# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_convolution_mean_0(input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 1156
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_indices
    x0 = x_indices % 16
    input_data1 = tl.load(input_ptr1 + (x0), None, eviction_policy='evict_last')
    accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r2 = r_indices
        input_data0 = tl.load(input_ptr0 + (r2 + (1156 * x3)), r_mask, eviction_policy='evict_first', other=0.0)
        combined_data = input_data0 + input_data1
        broadcasted_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
        updated_sum = accumulated_sum + broadcasted_data
        accumulated_sum = tl.where(r_mask, updated_sum, accumulated_sum)
    
    final_sum = tl.sum(accumulated_sum, 1)[:, None]
    tl.store(output_ptr0 + (x3), final_sum, None)