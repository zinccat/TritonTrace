# From: 78_ConvTranspose3d_Max_Max_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    reduction_z = reduction_index
    input_x = (input_index % kernel_size_x)
    input_y = input_index // kernel_size_x
    input_flat_index = input_index
    
    divisor_z = triton_helpers.div_floor_integer((-4) + kernel_size_z, 3)
    divisor_y = triton_helpers.div_floor_integer((-4) + kernel_size_y, 3)
    
    load_offset = (
        reduction_z + input_x + 16 * input_y +
        reduction_z * divisor_z * divisor_z +
        reduction_z * divisor_y +
        2 * reduction_z * divisor_z +
        16 * input_y * divisor_z * divisor_z +
        16 * input_y * divisor_y +
        32 * input_y * divisor_z +
        reduction_z * divisor_z * divisor_z * divisor_y +
        2 * reduction_z * divisor_y * divisor_z +
        16 * input_y * divisor_z * divisor_z * divisor_y +
        32 * input_y * divisor_y * divisor_z
    )
    
    tmp0 = tl.load(in_ptr0 + load_offset, input_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(input_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (input_flat_index), tmp4, input_mask)