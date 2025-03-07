# From: 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_convolution_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 4096
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_indices
    x0 = x_indices % 16
    input1_values = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r2 = r_indices
        input0_values = tl.load(in_ptr0 + (r2 + (4096 * x3)), r_mask, eviction_policy='evict_first', other=0.0)
        combined_values = input0_values + input1_values
        scaling_factor = 0.5
        scaled_values = combined_values * scaling_factor
        broadcasted_values = tl.broadcast_to(scaled_values, [XBLOCK, RBLOCK])
        accumulated_result = accumulated_result + broadcasted_values
        accumulated_result = tl.where(r_mask, accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    normalization_factor = 4096.0
    normalized_result = summed_result / normalization_factor
    final_result = normalized_result / 1.0
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), final_result, None)