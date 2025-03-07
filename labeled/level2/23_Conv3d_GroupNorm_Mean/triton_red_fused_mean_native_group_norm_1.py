# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_mean_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 50400
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices
    x_modulo = x_indices % 4
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_index = r_indices
        temp0 = tl.load(input_ptr0 + (r_index + (50400 * x_channel)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp1 = tl.load(input_ptr1 + ((2 * x_channel) + (r_index // 25200)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp3 = tl.load(input_ptr2 + ((2 * x_channel) + (r_index // 25200)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp10 = tl.load(input_ptr3 + ((4 * x_modulo) + (r_index // 12600)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp12 = tl.load(input_ptr4 + ((4 * x_modulo) + (r_index // 12600)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        temp2 = temp0 - temp1
        divisor = 25200.0
        temp5 = temp3 / divisor
        epsilon = 1e-05
        temp7 = temp5 + epsilon
        temp8 = tl.extra.cuda.libdevice.rsqrt(temp7)
        temp9 = temp2 * temp8
        temp11 = temp9 * temp10
        temp13 = temp11 + temp12
        temp14 = tl.broadcast_to(temp13, [XBLOCK, RBLOCK])
        temp16 = temp_sum + temp14
        temp_sum = tl.where(r_mask & x_mask, temp16, temp_sum)

    temp15 = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_channel), temp15, x_mask)