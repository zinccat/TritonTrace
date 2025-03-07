# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_min_native_group_norm_1per_fused_min_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input, 
    output_ptr_min, output_ptr_argmin, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]
    
    mean = tl.load(input_ptr_mean + (r_index + 256 * x_index), None)
    var = tl.load(input_ptr_var + (8 * x_index + (r_index // 32)), None, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (8 * x_index + (r_index // 32)), None, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (r_index), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (r_index), None, eviction_policy='evict_last')

    normalized_input = (input_data - mean) * (1 / (tl.sqrt(var / 32.0 + 1e-05)))
    scaled_input = normalized_input * gamma + beta

    broadcast_scaled_input = tl.broadcast_to(scaled_input, [RBLOCK])
    min_value = triton_helpers.promote_to_tensor(triton_helpers.min2(broadcast_scaled_input, 0))
    argmin_index = triton_helpers.promote_to_tensor(triton_helpers.min_with_index(broadcast_scaled_input, r_index, 0)[1])

    tl.store(output_ptr_min + (x_index), min_value, None)
    tl.store(output_ptr_argmin + (x_index), argmin_index, None)