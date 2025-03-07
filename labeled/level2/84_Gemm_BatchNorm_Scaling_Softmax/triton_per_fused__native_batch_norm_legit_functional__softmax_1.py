# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_1(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_min, input_ptr_max, 
    output_ptr, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]
    
    mean = tl.load(input_ptr_mean + (r_index + (512 * x_index)), None)
    variance = tl.load(input_ptr_var + (r_index), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (r_index), None, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (r_index), None, eviction_policy='evict_last')
    min_val = tl.load(input_ptr_min + (0))
    max_val = tl.load(input_ptr_max + (0))

    min_broadcast = tl.broadcast_to(min_val, [RBLOCK])
    max_broadcast = tl.broadcast_to(max_val, [RBLOCK])

    normalized_input = mean - variance
    scaled_input = normalized_input * scale
    shifted_input = scaled_input * shift + shift

    relu_mask = tl.where(max_broadcast >= 0.0, 1.0, -1.0)
    relu_applied = shifted_input * relu_mask

    relu_applied_broadcast = tl.broadcast_to(relu_applied, [RBLOCK])
    max_value = triton_helpers.promote_to_tensor(tl.max2(relu_applied_broadcast, 0))
    shifted_input_stable = relu_applied - max_value

    relu_mask_broadcast = tl.broadcast_to(relu_mask, [RBLOCK])
    min_val_broadcast = tl.broadcast_to(min_val, [RBLOCK])
    adjusted_min = relu_mask_broadcast * min_val_broadcast

    exponent_input = shifted_input_stable * adjusted_min
    exp_values = tl.math.exp(exponent_input)
    exp_values_broadcast = tl.broadcast_to(exp_values, [RBLOCK])

    sum_exp = triton_helpers.promote_to_tensor(tl.sum(exp_values_broadcast, 0))
    softmax_output = exp_values / sum_exp

    tl.store(output_ptr + (r_index + (512 * x_index)), softmax_output, None)