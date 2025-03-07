# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_avg_pool2d_gelu_max_mul_0(
    input_ptr, output_ptr_avg, output_ptr_max_val, output_ptr_max_idx, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices

    input_data_0 = tl.load(input_ptr + (4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_data_1 = tl.load(input_ptr + (1 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_data_2 = tl.load(input_ptr + (2 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_data_3 = tl.load(input_ptr + (3 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)

    sum_01 = input_data_1 + input_data_0
    sum_23 = input_data_2 + sum_01
    sum_0123 = input_data_3 + sum_23

    scale_factor = 0.25
    scaled_sum = sum_0123 * scale_factor

    gelu_factor = 0.5
    scaled_gelu = scaled_sum * gelu_factor

    erf_factor = 0.7071067811865476
    erf_input = scaled_sum * erf_factor

    erf_result = tl.extra.cuda.libdevice.erf(erf_input)

    erf_adjusted = erf_result + 1.0
    gelu_output = scaled_gelu * erf_adjusted

    final_output = gelu_output * 2.0
    broadcast_output = tl.broadcast_to(final_output, [XBLOCK, RBLOCK])

    masked_output = tl.where(x_mask, broadcast_output, float("-inf"))
    max_values = triton_helpers.max2(masked_output, 1)[:, None]

    broadcast_r_indices = tl.broadcast_to(r_indices, masked_output.shape)
    max_values_val, max_values_idx = triton_helpers.max_with_index(masked_output, broadcast_r_indices, 1)

    tl.store(output_ptr_avg + (r1 + 64 * x0), scaled_sum, x_mask)
    tl.store(output_ptr_max_val + (x0), max_values, x_mask)
    tl.store(output_ptr_max_idx + (x0), max_values_idx[:, None], x_mask)