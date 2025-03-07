# From: 45_Average_Pooling_2D

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 7398400
    program_id = tl.program_id(0)
    xoffset = program_id * XBLOCK
    xindices = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindices < xnumel

    x_col = xindices % 85
    x_row = (xindices // 85) % 85
    x_depth = xindices // 7225
    x_linear_index = xindices

    # Load input values with eviction policy
    input_val_0 = tl.load(in_ptr0 + ((3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_1 = tl.load(in_ptr0 + (1 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_2 = tl.load(in_ptr0 + (2 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_3 = tl.load(in_ptr0 + (256 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_4 = tl.load(in_ptr0 + (257 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_5 = tl.load(in_ptr0 + (258 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_6 = tl.load(in_ptr0 + (512 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_7 = tl.load(in_ptr0 + (513 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')
    input_val_8 = tl.load(in_ptr0 + (514 + (3 * x_col) + (768 * x_row) + (65536 * x_depth)), xmask, eviction_policy='evict_last')

    # Sum the loaded values
    sum_val_1 = input_val_1 + input_val_0
    sum_val_2 = input_val_2 + sum_val_1
    sum_val_3 = input_val_3 + sum_val_2
    sum_val_4 = input_val_4 + sum_val_3
    sum_val_5 = input_val_5 + sum_val_4
    sum_val_6 = input_val_6 + sum_val_5
    sum_val_7 = input_val_7 + sum_val_6
    sum_val_8 = input_val_8 + sum_val_7

    # Calculate average
    average_value = sum_val_8 * 0.1111111111111111

    # Store the result
    tl.store(out_ptr0 + (x_linear_index), average_value, xmask)