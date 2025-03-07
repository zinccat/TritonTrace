# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_2(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_num_elements = 368
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_indices // 16
    x0 = (x_indices % 16)
    bias_value = tl.load(input_ptr1 + (0))
    bias_broadcast = tl.broadcast_to(bias_value, [XBLOCK, RBLOCK])
    accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_indices

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < r_num_elements
        r2 = r_indices
        divisor = triton_helpers.div_floor_integer(22 + ((-1) * kernel_size0) + kernel_size0 * kernel_size1 + ((-1) * kernel_size0 * kernel_size2 * kernel_size2) + 2 * kernel_size0 * kernel_size2 + kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size0 * kernel_size1 * kernel_size2), 23)
        tmp0 = r2 + x1 * divisor
        tmp1 = ((-1) * kernel_size0) + kernel_size0 * kernel_size1 + ((-1) * kernel_size0 * kernel_size2 * kernel_size2) + 2 * kernel_size0 * kernel_size2 + kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size0 * kernel_size1 * kernel_size2)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(input_ptr0 + (((-1) * x0) + ((-1) * (((r2 + x1 * divisor) // ((-1) + kernel_size2)) % ((-1) + kernel_size2)))) + ((-16) * (((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0)) + kernel_size1 * x0 + kernel_size2 * (((((r2 + x1 * divisor) // ((-1) + kernel_size2)) % ((-1) + kernel_size2))) + kernel_size2 * kernel_size2 * (((((r2 + x1 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1))) + ((-1) * x0 * kernel_size2 * kernel_size2) + ((-16) * kernel_size2 * kernel_size2 * (((((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0))) + ((-2) * kernel_size2 * (((((r2 + x1 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1)))) + 2 * kernel_size2 * x0 + 16 * kernel_size1 * (((((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0)) + 32 * kernel_size2 * (((((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0)) + kernel_size1 * x0 * kernel_size2 * kernel_size2 + ((-32) * kernel_size1 * kernel_size2 * (((((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0))) + ((-2) * kernel_size1 * kernel_size2 * x0) + 16 * kernel_size1 * kernel_size2 * kernel_size2 * (((((r2 + x1 * divisor) // ((-1) + kernel_size1 + ((-1) * kernel_size2 * kernel_size2) + 2 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + ((-2) * kernel_size1 * kernel_size2))) % kernel_size0)) + (((r2 + x1 * divisor) % ((-1) + kernel_size2))) + ((((r2 + x1 * divisor) // (1 + kernel_size2 * kernel_size2 + ((-2) * kernel_size2))) % ((-1) + kernel_size1))))), r_mask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp3 * bias_broadcast
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = accumulator + tmp9
        accumulator = tl.where(r_mask & x_mask, tmp11, accumulator)

    result = tl.sum(accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x3), result, x_mask)