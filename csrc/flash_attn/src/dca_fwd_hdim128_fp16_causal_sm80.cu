// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "dca_fwd_launch_template.h"

template<>
void run_dca_fwd_<cutlass::half_t, 128, true>(Flash_dca_fwd_params &params, cudaStream_t stream) {
    run_dca_fwd_hdim128<cutlass::half_t, true>(params, stream);
}
