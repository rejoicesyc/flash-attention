# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func
from flash_attn import flash_attn_func, flash_attn_varlen_func

import nvtx
#from triton_dca import triton_dca
from vllm.attention.ops.triton_dca_bthd import triton_dca
from vllm.attention.ops.triton_dca_bhtd import triton_dca_bhtd

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None


try:
    import xformers.ops as xops
except ImportError:
    xops = None


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


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

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


repeats = 50
device = 'cuda'
dtype = torch.float16

#bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
#bs_seqlen_vals = [(32, 512), (16, 1024), (8, 4096), (4, 8192), (2, 16384), (1, 32768)]
bs_seqlen_vals = [(1, 16384), (1, 32768), (1, 64 * 1024), (1, 128 * 1024)]
# bs_seqlen_vals = [(1, 32768), (1, 128 * 1024), (1, 512 * 1024)]
# bs_seqlen_vals = [(1, 128 * 1024), (1, 512 * 1024)]
causal_vals = [True]
headdim_vals = [64]
#headdim_vals = [128]
dim = 2048
dropout_p = 0.0

methods = (["Flash2"]#, "Pytorch"]
           + (["Triton"] if attention_triton is not None else [])
           + ["triton_dca_bhtd"]
           + ['dca']
        )

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

chunk_lens = [2560]
local_size = 0

for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            for chunk_len in chunk_lens:
                config = (causal, headdim, batch_size, seqlen)
                nheads = dim // headdim

                qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=False)
                f = time_fwd(
                    flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                )
                time_f[config, "Flash2"] = f

                # q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                #                         requires_grad=False) for _ in range(3)]
                # f = time_fwd(
                #     flash_attn_func, q, k, v, dropout_p, causal, repeats=repeats, verbose=False
                # )
                # time_f[config, "Flash2"] = f

                if 'triton_dca_bhtd' in methods:
                    q, q_succ, q_inter, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                            requires_grad=False) for _ in range(5)]
                    f = time_fwd(
                        triton_dca_bhtd, q, q_succ, q_inter, k, v, True, None, 0., chunk_len, repeats=repeats, verbose=False
                    )
                    time_f[config, 'triton_dca_bhtd'] = f

                if 'dca' in methods:
                    q, q_succ, q_inter, k, v = [torch.randn(batch_size * seqlen, nheads, headdim, device=device, dtype=dtype,
                                            requires_grad=False) for _ in range(5)]
                
                    f = time_fwd(
                        _bruteforce_dynamic_chunk_flash_attn_func, q, q_succ, q_inter, k, v, None, None, chunk_len, 0,
                        0, None, seqlen, repeats=repeats, verbose=False
                    )
                    time_f[config, 'dca'] = f

                if 'Triton' in methods:
                    q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                        requires_grad=False) for _ in range(3)]
                    # Try both values of sequence_parallel and pick the faster one
                    f = time_fwd(
                        attention_triton, q, k, v, causal, headdim**(-0.5),
                        False, repeats=repeats, verbose=False
                    )

                    time_f[config, "Triton"] = f

                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
                for method in methods:
                    #time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )

                    print(
                        f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    )
