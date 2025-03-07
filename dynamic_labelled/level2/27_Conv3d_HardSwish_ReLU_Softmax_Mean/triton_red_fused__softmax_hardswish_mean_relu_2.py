# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_hardswish_mean_relu_2red_fused__softmax_hardswish_mean_relu_2(
    output_ptr, input_ptr0, input_ptr1, input_ptr2, kernel_size0, kernel_size1, kernel_size2, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x1 = x_index // 16
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r2 = r_index

        input_val0 = tl.load(
            input_ptr0 + (r2 + ((-8) * x3) + ((-2) * x3 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x3 + 8 * kernel_size1 * x3 + kernel_size0 * x3 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x3)),
            r_mask & x_mask, eviction_policy='evict_first', other=0.0
        )

        input_val1 = tl.load(
            input_ptr1 + (r2 + ((-8) * x1) + ((-2) * x1 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x1 + 8 * kernel_size1 * x1 + kernel_size0 * x1 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x1)),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        input_val2 = tl.load(
            input_ptr2 + (r2 + ((-8) * x1) + ((-2) * x1 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x1 + 8 * kernel_size1 * x1 + kernel_size0 * x1 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x1)),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        bias = 3.0
        biased_input = input_val0 + bias
        lower_bound = 0.0
        upper_bound = 6.0

        clipped_input = triton_helpers.minimum(triton_helpers.maximum(biased_input, lower_bound), upper_bound)
        hard_swish = clipped_input * input_val0 * 0.16666666666666666

        max_val = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), hard_swish)
        exp_input = tl.math.exp(max_val - input_val1)
        softmax_output = exp_input / input_val2

        broadcasted_output = tl.broadcast_to(softmax_output, [XBLOCK, RBLOCK])
        temp_sum = tl.where(r_mask & x_mask, temp_sum + broadcasted_output, temp_sum)

    sum_over_r = tl.sum(temp_sum, 1)[:, None]
    kernel_size2_float = kernel_size2.to(tl.float32)
    mean_output = sum_over_r / kernel_size2_float

    tl.debug_barrier()
    tl.store(output_ptr + (x3), mean_output, x_mask)