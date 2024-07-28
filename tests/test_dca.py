import math
from typing import Any, Dict, List, Optional, Type

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    dca_varlen_func,
    dca_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    flash_attn_func,
)
from flash_attn.layers.rotary import apply_rotary_emb

from tests.test_flash_attn import (
    generate_qkv,
    attention_ref,
    _generate_block_kvcache,
    generate_random_padding_mask,
    attn_bias_from_alibi_slopes
)


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


"""
reference implement from https://github.com/vllm-project/vllm/pull/6139
"""

def _bruteforce_dynamic_chunk_flash_attn_func(
    q,
    q_succ,
    q_inter,
    k,
    v,
    block_table,
    softmax_scale,
    chunk_size,
    local_size,
    original_max_position_embeddings,
    current_prefill_original_seq_lens_tensor,
    k_length,
):

    def do_flash_attn(
        query_states,
        key_states,
        value_states,
        causal=True,
        block_table=None,
        max_seqlen_k=None,
    ):
        if max_seqlen_k is None:
            max_seqlen_k = key_states.shape[0]

        output, softmax_lse, _ = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            softmax_scale=softmax_scale,
            cu_seqlens_q=torch.tensor(
                [0, query_states.shape[0]],
                dtype=torch.int32,
                device=query_states.device,
            ),
            max_seqlen_q=query_states.shape[0],
            cu_seqlens_k=torch.tensor(
                [0, max_seqlen_k],
                dtype=torch.int32,
                device=query_states.device,
            ),
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            block_table=block_table,
            return_attn_probs=True,
        )
        return output, softmax_lse

    def merge_attn_outputs(flash_results):
        attn_outputs_all = []
        for flash_per_chunk in flash_results:
            if len(flash_per_chunk) == 1:
                attn_outputs_all.append(flash_per_chunk[0][0])
                continue
            attn_outputs = torch.stack([
                flash_attn_output[0]
                for flash_attn_output in flash_per_chunk
            ])
            logits = torch.stack([
                flash_attn_output[1]
                for flash_attn_output in flash_per_chunk
            ]).to(torch.float32)
            max_logits = torch.max(logits, dim=0).values
            stable_logits = logits - max_logits.unsqueeze(0)
            lse_s = torch.exp(stable_logits).detach()
            lse_sum = torch.sum(lse_s, dim=0)
            lse_s /= lse_sum
            attn_outputs *= lse_s.unsqueeze(-1).transpose(2, 3).squeeze(1)
            attn_outputs_all.append(attn_outputs.sum(dim=0))
        return torch.cat(attn_outputs_all, dim=0)

    def get_block(begin, end):
        return block_table[:,
                            begin // block_size:(end - 1) // block_size + 1]

    flash_results = []
    chunk_len = chunk_size - local_size
    if block_table is not None:
        block_size = v.shape[1]
        if chunk_len % block_size != 0:
            raise ValueError("chunk_len must be divisible by block_size.")
    else:
        block_size = 1

    if original_max_position_embeddings > 0:
        mscale = max(
            0.1 * (current_prefill_original_seq_lens_tensor[0] /
                    original_max_position_embeddings).log() + 1.0,
            1.0,
        )
        softmax_scale = softmax_scale * mscale

    begin = k_length - q.shape[0]

    while begin < k_length:
        flash_per_chunk = []

        prev_chunk_end_pos = (begin // chunk_len) * chunk_len
        next_chunk_end_pos = prev_chunk_end_pos + chunk_len
        end = min(next_chunk_end_pos, k_length)
        qbegin = begin - (k_length - q.shape[0])
        qend = end - (k_length - q.shape[0])

        q_states_intra = q[qbegin:qend]
        if block_table is not None:
            block_table_intra = get_block(prev_chunk_end_pos, end)
            flash_result = do_flash_attn(
                q_states_intra,
                k,
                v,
                block_table=block_table_intra,
                max_seqlen_k=end - prev_chunk_end_pos,
            )
        else:
            # print(f'intra flash q[{qbegin}:{qend}] kv[{prev_chunk_end_pos}:{end}]')
            k_states_intra = k[prev_chunk_end_pos:end]
            v_states_intra = v[prev_chunk_end_pos:end]
            flash_result = do_flash_attn(q_states_intra, k_states_intra,
                                            v_states_intra)
        flash_per_chunk.append(flash_result)

        if prev_chunk_end_pos - chunk_len >= 0:
            q_states_succ = q_succ[qbegin:qend]
            if block_table is not None:
                block_table_succ = get_block(
                    prev_chunk_end_pos - chunk_len, prev_chunk_end_pos)
                flash_result = do_flash_attn(
                    q_states_succ,
                    k,
                    v,
                    False,
                    block_table=block_table_succ,
                    max_seqlen_k=chunk_len,
                )
            else:
                k_states_succ = k[prev_chunk_end_pos -
                                    chunk_len:prev_chunk_end_pos]
                v_states_succ = v[prev_chunk_end_pos -
                                    chunk_len:prev_chunk_end_pos]
                flash_result = do_flash_attn(q_states_succ, k_states_succ,
                                                v_states_succ, False)
            flash_per_chunk.append(flash_result)

        if prev_chunk_end_pos - chunk_len * 2 >= 0:
            q_states_inter = q_inter[qbegin:qend]

            if block_table is not None:
                block_table_inter = get_block(
                    0, prev_chunk_end_pos - chunk_len)
                flash_result = do_flash_attn(
                    q_states_inter,
                    k,
                    v,
                    False,
                    block_table=block_table_inter,
                    max_seqlen_k=prev_chunk_end_pos - chunk_len,
                )
            else:
                k_states_inter = k[:prev_chunk_end_pos - chunk_len]
                v_states_inter = v[:prev_chunk_end_pos - chunk_len]
                flash_result = do_flash_attn(q_states_inter,
                                                k_states_inter,
                                                v_states_inter, False)
            flash_per_chunk.append(flash_result)

        begin = end
        flash_results.append(flash_per_chunk)

    attn_output = merge_attn_outputs(flash_results)

    return attn_output

def _bruteforce_dynamic_chunk_flash_attn_varlen_func(
        q,
        q_succ,
        q_inter,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_table,
        chunk_size,
        local_size,
        original_max_position_embeddings,
        prefill_original_seq_lens_tensor,
    ):

        if alibi_slopes is not None:
            raise ValueError(
                "Native Dynamic Chunk Attention does not support alibi_slopes")
        if not causal:
            raise ValueError(
                "Native Dynamic Chunk Attention does not support causal=False")
        if window_size != (-1, -1):
            raise ValueError(
                "Native Dynamic Chunk Attention does not support window_size")

        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().tolist()

        all_outputs = []
        for i in range(0, len(cu_seqlens_q_cpu) - 1):
            qs = cu_seqlens_q_cpu[i]
            qe = cu_seqlens_q_cpu[i:i + 2][-1]
            ks = cu_seqlens_k_cpu[i]
            ke = cu_seqlens_k_cpu[i:i + 2][-1]

            current_q = q[qs:qe]
            current_q_succ = q_succ[qs:qe]
            current_q_inter = q_inter[qs:qe]
            if block_table is None:
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
            else:
                current_block_table = block_table[i:i + 1]
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
                current_k = k
                current_v = v

            if current_q.shape[0] == 0:
                continue
            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q.shape[0], current_q.shape[1], v.shape[2]),
                        device=q.device,
                        dtype=q.dtype,
                    ))
                continue

            current_output = _bruteforce_dynamic_chunk_flash_attn_func(
                current_q,
                current_q_succ,
                current_q_inter,
                current_k,
                current_v,
                current_block_table,
                softmax_scale,
                chunk_size,
                local_size,
                original_max_position_embeddings,
                current_prefill_original_seq_lens_tensor,
                ke - ks,
            )
            all_outputs.append(current_output)

        return torch.cat(all_outputs, dim=0)

def _bruteforce_dynamic_chunk_pageattention_forward_decode(
    query: torch.Tensor,
    query_succ: torch.Tensor,
    query_inter: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: Optional[torch.Tensor],
    chunk_size: int,
    local_size: int,
    original_max_position_embeddings: int,
):
    assert causal
    batch_size = block_table.shape[0]
    # print(value_cache.shape)
    block_size = value_cache.shape[1]
    chunk_len = chunk_size - local_size
    if chunk_len % block_size != 0:
        raise ValueError(f"chunk_len must be divisible by block_size. With {chunk_len}/{block_size}")
    chunk_num_curr = (cache_seqlens - 1) // chunk_len

    if original_max_position_embeddings > 0:
        mscale = (
            0.1 *
            torch.log(cache_seqlens / original_max_position_embeddings) +
            1.0).clip(min=1)
        query = (query * mscale.view(-1, 1, 1, 1)).to(
            query.dtype
        )  # possible for numerical issue, need to fused in the kernel
        query_succ = (query_succ * mscale.view(-1, 1, 1, 1)).to(
            query.dtype)
        query_inter = (query_inter * mscale.view(-1, 1, 1, 1)).to(
            query.dtype)

    outputs_list = []
    softmax_lses_list = []

    # intra-attention
    seq_lens_intra = cache_seqlens - chunk_num_curr * chunk_len
    max_seq_len_intra = seq_lens_intra.max().item()
    block_table_intra = torch.zeros(
        batch_size,
        (max_seq_len_intra - 1) // block_size + 1,
        dtype=block_table.dtype,
        device=block_table.device,
    )
    for i in range(batch_size):
        st = chunk_num_curr[i] * chunk_len // block_size
        ed = min(
            st + (max_seq_len_intra - 1) // block_size + 1,
            (cache_seqlens[i] - 1) // block_size + 1,
        )
        block_table_intra[i, :ed - st] = block_table[i, st:ed]
    intra_output, intra_softmax_lse = (
        _pagedattention_forward_decode_with_exp_sums(
            query,
            key_cache,
            value_cache,
            block_table_intra,
            seq_lens_intra,
            softmax_scale,
            alibi_slopes,
            causal=False,
        ))
    outputs_list.append(intra_output)
    softmax_lses_list.append(intra_softmax_lse)

    # succ-attention
    seq_lens_succ = (chunk_num_curr -
                     (chunk_num_curr - 1).clip(min=0)) * chunk_len
    max_seq_len_succ = seq_lens_succ.max().item()
    if max_seq_len_succ:
        block_table_succ = torch.zeros(
            batch_size,
            (max_seq_len_succ - 1) // block_size + 1,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        for i in range(batch_size):
            st = ((chunk_num_curr[i] - 1).clip(min=0) * chunk_len //
                  block_size)
            ed = min(
                st + (max_seq_len_succ - 1) // block_size + 1,
                (cache_seqlens[i] - 1) // block_size + 1,
            )
            block_table_succ[i, :ed - st] = block_table[i, st:ed]
        succ_output, succ_softmax_lse = (
            _pagedattention_forward_decode_with_exp_sums(
                query_succ,
                key_cache,
                value_cache,
                block_table_succ,
                seq_lens_succ,
                softmax_scale,
                alibi_slopes,
                causal=False,
            ))
        outputs_list.append(succ_output)
        softmax_lses_list.append(succ_softmax_lse)

    # inter-attention
    seq_lens_inter = (chunk_num_curr - 1).clip(min=0) * chunk_len
    max_seq_len_inter = seq_lens_inter.max().item()
    if max_seq_len_inter:
        inter_output, succ_softmax_lse = (
            _pagedattention_forward_decode_with_exp_sums(
                query_inter,
                key_cache,
                value_cache,
                block_table[:, :max_seq_len_inter],
                seq_lens_inter,
                softmax_scale,
                alibi_slopes,
                causal=False,
            ))
        outputs_list.append(inter_output)
        softmax_lses_list.append(succ_softmax_lse)

    outputs = torch.stack(outputs_list, dim=0)
    del outputs_list
    softmax_lses = torch.stack(softmax_lses_list, dim=0).to(torch.float32)
    del softmax_lses_list

    max_logits = torch.max(softmax_lses, dim=0).values
    stable_logits = softmax_lses - max_logits.unsqueeze(0)
    lse_s = torch.exp(stable_logits).detach()
    lse_sum = torch.sum(lse_s, dim=0)
    lse_s /= lse_sum
    outputs *= lse_s.unsqueeze(-1).transpose(2, 3)

    return outputs.sum(0)


def _pagedattention_forward_decode_with_exp_sums(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
):

    # out, softmax_lse = flash_attn_with_kvcache(
    # print('flash_attn_with_kvcache', query.shape, key_cache.shape, value_cache.shape)
    out, softmax_lse = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        # return_softmax=False,
    )
    cache_seqlens_cpu = cache_seqlens.cpu()
    for i in range(cache_seqlens.shape[0]):
        if cache_seqlens_cpu[i] == 0:
            softmax_lse[i].fill_(-float("inf"))
            out[i].fill_(0)

    return out, softmax_lse


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k",
    [
        (14, 1024, 1024),
        (1, 13277, 13277),
    ],
)
@pytest.mark.parametrize(
    "nheads_q, nheads_k", [
        (8, 1), # test gqa
        (8, 8),
    ]
)
# TODO: add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
@pytest.mark.parametrize("paged_kv_block_size", [None])
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 128)])
def test_dca_varlen_causal(
    batch_size, seqlen_q, seqlen_k, nheads_q, nheads_k, d, local, paged_kv_block_size, dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM

    assert seqlen_q == seqlen_k, "this test is for prefill"
    device = "cuda"
    causal = True
    # set seed
    torch.random.manual_seed(0)
    # nheads = 8
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads_q, d, device=device, dtype=dtype, requires_grad=False)

    if paged_kv_block_size is None:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False
        )
        block_table = None
    else:
        k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    out_unpad = dca_varlen_func(
        q_unpad,
        k_unpad if paged_kv_block_size is None else k_cache_paged,
        v_unpad if paged_kv_block_size is None else v_cache_paged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out = output_pad_fn(out_unpad)
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_splits", [1, 0])
# @pytest.mark.parametrize("num_splits", [1])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("new_kv", [False])
# @pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [False])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [False])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("paged_kv_block_size", [256])
# @pytest.mark.parametrize("paged_kv_block_size", [256, 512])
# @pytest.mark.parametrize("paged_kv_block_size", [None])
# @pytest.mark.parametrize("has_leftpad", [False, True])
@pytest.mark.parametrize("has_leftpad", [False])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 128),
        # (1, 339),
        # (3, 1024),
        # (64, 800),
        # (64, 256),
        # (3, 799),
        # (64, 2048),
        # (16, 20000),
        (1, 1001),
        (3, 128 * 1024),
        # (128, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_dca_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    has_leftpad,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    mha_type,
    num_splits,
    dtype,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if has_batch_idx and paged_kv_block_size is not None:
        pytest.skip()
    if has_leftpad and paged_kv_block_size is not None:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (
            (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
            if new_kv
            else (seqlen_k + 1)
        ),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    if has_leftpad:
        cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                   if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                   for i in range(batch_size)])
    else:
        cache_leftpad = None
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_leftpad:
        key_padding_mask = torch.logical_and(
            key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
        )
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal, key_leftpad=cache_leftpad
        )
    else:
        alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = (
            torch.rand(
                seqlen_k if paged_kv_block_size is None else num_blocks * paged_kv_block_size,
                rotary_dim // 2,
                device=device,
            )
            * 2
            * math.pi
        )
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    out = dca_with_kvcache(
        q,
        k_cache if paged_kv_block_size is None else k_cache_paged,
        v_cache if paged_kv_block_size is None else v_cache_paged,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
    )
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        key_leftpad=cache_leftpad,
    )
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        key_leftpad=cache_leftpad,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = (
                k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
            )
            v_cache_select = (
                v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5
