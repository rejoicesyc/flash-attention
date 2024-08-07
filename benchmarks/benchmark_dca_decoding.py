# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import nvtx

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_with_kvcache, flash_dca_with_kvcache

from typing import Any, Dict, List, Optional, Type


NUM_BLOCKS = 1024
PARTITION_SIZE = 512

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

@nvtx.annotate('dca_decode', color='red')
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
        return_softmax_lse=True,
    )
    cache_seqlens_cpu = cache_seqlens.cpu()
    for i in range(cache_seqlens.shape[0]):
        if cache_seqlens_cpu[i] == 0:
            softmax_lse[i].fill_(-float("inf"))
            out[i].fill_(0)

    return out, softmax_lse


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen,
                         dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full(
            (seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


def time_fwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    time_f = benchmark_forward(
        fn,
        *inputs,
        repeats=repeats,
        desc=desc,
        verbose=verbose,
        amp=amp,
        amp_dtype=amp_dtype,
        **kwinputs,
    )
    return time_f[1].mean


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks


repeats = 30
device = 'cuda'
dtype = torch.float16

# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
seqlen_qk = [
    # (1, 128),
    # (1, 339),
    # (3, 1024),
    # (64, 800),
    # (64, 256),
    # (3, 799),
    # (64, 2048),
    # (16, 20000),
    (1, 32 * 1024),
    (1, 64 * 1024),
    (1, 128 * 1024),
    (1, 512 * 1024),
    (1, 1024 * 1024)
    # (128, 128),
]
paged_kv_block_sizes = [1024]
causal_vals = [False]
headdim_vals = [128]
new_kvs = [False]
dim = 2048
dropout_p = 0.0
num_kv_heads = 8

methods = [
    'flash_attn_with_kvcache',
    'flash_dca_with_kvcache',
    # 'paged_attention_v2',
    'dca_decode',
]
chunk_infos = [(8 * 1024, 0), (32 * 1024, 0)]

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for (chunk_size, local_size), causal, headdim, (seqlen_q, seqlen_k), new_kv, paged_kv_block_size in product(
    chunk_infos, causal_vals, headdim_vals, seqlen_qk, new_kvs, paged_kv_block_sizes
):
    if seqlen_q > seqlen_k and new_kv:
        continue
    if (chunk_size - local_size) * 3 > max(seqlen_q, seqlen_k):
        continue

    batch_size = 1
    config = (causal, headdim, batch_size, seqlen_q,
              seqlen_k, new_kv, paged_kv_block_size)
    nheads = dim // headdim
    batch_size_cache = 1
    nheads_k = 8
    local = False
    seqlen_new_eq_seqlen_q = seqlen_q
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(
        1, seqlen_q + 1, (1,)).item()
    rotary_dim = headdim

    q = torch.randn(batch_size, seqlen_q, nheads,
                    headdim, device=device, dtype=dtype)
    q_succ = torch.randn(batch_size, seqlen_q, nheads,
                         headdim, device=device, dtype=dtype)
    q_inter = torch.randn(batch_size, seqlen_q, nheads,
                          headdim, device=device, dtype=dtype)
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k,
                        headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k,
                        headdim, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k,
                              nheads_k, headdim, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k,
                              nheads_k, headdim, device=device, dtype=dtype)
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
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, headdim, device, dtype
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local)
         and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [k_cache.shape[1]],
        dtype=torch.int32,
        device=q.device,
    )

    if 'flash_attn_with_kvcache' in methods:

        # print('flash_attn_with_kvcache', q.shape, k_cache.shape, v_cache.shape)
        with nvtx.annotate("flash_decoding", color="green"):
            f = time_fwd(
                flash_attn_with_kvcache,
                q,
                k_cache if paged_kv_block_size is None else k_cache_paged,
                v_cache if paged_kv_block_size is None else v_cache_paged,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                softmax_scale=None,
                alibi_slopes=None,
                causal=causal,
                # return_softmax=False,
                repeats=repeats, 
                verbose=False
            )
            time_f[config, 'flash_attn_with_kvcache'] = f

    if 'flash_dca_with_kvcache' in methods:
        f = time_fwd(
            flash_dca_with_kvcache,
            q,
            q_succ,
            q_inter,
            k_cache if paged_kv_block_size is None else k_cache_paged,
            v_cache if paged_kv_block_size is None else v_cache_paged,
            chunk_size,
            local_size,
            k,
            v,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            cache_leftpad=None,
            block_table=block_table,
            causal=causal,
            window_size=(-1, -1),
            rotary_interleaved=False,
            alibi_slopes=None,
            num_splits=0,
        )
        time_f[config, 'flash_dca_with_kvcache'] = f


    # if 'paged_attention_v2' in methods:

    #     # Prepare for the paged attention kernel.
    #     output = torch.empty_like(q)
    #     ops.paged_attention_v2(
    #         output,
    #         exp_sums,
    #         max_logits,
    #         tmp_output,
    #         q,
    #         k_cache,
    #         v_cache,
    #         num_kv_heads,
    #         None,
    #         block_table,
    #         cache_seqlens,
    #         block_size,
    #         max_seq_len,
    #         None,
    #         dtype,
    #         None,
    #     )

    if 'dca_decode' in methods:
        # print(q.shape, k_cache.shape, v_cache.shape)            
        f = time_fwd(
            _bruteforce_dynamic_chunk_pageattention_forward_decode,
            q,
            q_succ,
            q_inter,
            k_cache if paged_kv_block_size is None else k_cache_paged,
            v_cache if paged_kv_block_size is None else v_cache_paged,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=None,
            causal=True,
            alibi_slopes=None,
            chunk_size=chunk_size,
            local_size=local_size,
            original_max_position_embeddings=32768,
        )
        time_f[config, 'dca_decode'] = f

    print(f"### chunk_len={chunk_size-local_size} causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, new_kv={new_kv}, paged_kv_block_size={paged_kv_block_size} ###")
    for method in methods:
        time_f[config, method] = time_f[config, method]
        # speed_f[config, method] = efficiency(
        #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
        #     time_f[config, method]
        # )

        print(
            # f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
            f"{method} \tfwd: {(time_f[config, method] * 10 ** 6):.4f} us, "
        )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)

