# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax__softmax_backward_data_clamp_mul_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x0 = (x_index % kernel_size0)
    x1 = x_index // kernel_size0
    x3 = x_index
    input_value0 = tl.load(input_ptr0 + (x0 + kernel_size1 * r2 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * x1 * kernel_size2 * kernel_size2), x_mask, eviction_policy='evict_last', other=0.0)
    input_value1 = tl.load(input_ptr1 + (x0 + kernel_size1 * r2 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * x1 * kernel_size2 * kernel_size2), x_mask, eviction_policy='evict_last', other=0.0)
    input_value2 = tl.load(input_ptr2 + (x3), x_mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (x3), x_mask, eviction_policy='evict_last')
    scale_factor = 2.0
    scaled_value0 = input_value0 * scale_factor
    clamp_min = 0.0
    clamped_value = triton_helpers.maximum(input_value1, clamp_min)
    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    exponent_base = clamped_value - input_value2
    exp_value = tl.math.exp(exponent_base)
    softmax_value = exp_value / input_value3
    result_value = scaled_value0 * softmax_value
    broadcasted_result = tl.broadcast_to(result_value, [XBLOCK, RBLOCK])
    masked_result = tl.where(x_mask, broadcasted_result, 0)
    summed_result = tl.sum(masked_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), summed_result, x_mask)