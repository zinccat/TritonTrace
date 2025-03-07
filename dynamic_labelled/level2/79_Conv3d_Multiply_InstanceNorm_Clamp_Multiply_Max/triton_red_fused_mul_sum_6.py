# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_6red_fused_mul_sum_6(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 336
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = x_index % 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + 
            4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + 
            kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size0 * kernel_size1 * kernel_size2), 
            21
        )

        tmp0 = r2 + x1 * divisor
        tmp1 = ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + 
               4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + 
               kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + ((-4) * kernel_size0 * kernel_size1 * kernel_size2)
        tmp2 = tmp0 < tmp1

        index0 = (((-128) * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)) + 
                  ((-8) * x0) + ((-2) * (((r2 + x1 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                  4 * ((((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                  kernel_size2 * ((((r2 + x1 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                  kernel_size2 * kernel_size2 * ((((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                  ((-32) * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)) + 
                  ((-4) * kernel_size2 * ((((r2 + x1 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)))) + 
                  ((-2) * x0 * kernel_size2 * kernel_size2) + 
                  4 * kernel_size1 * x0 + 
                  8 * kernel_size2 * x0 + 
                  64 * kernel_size1 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0) + 
                  128 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0) + 
                  kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                  ((-64) * kernel_size1 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0)) + 
                  ((-4) * kernel_size1 * kernel_size2 * x0) + 
                  16 * kernel_size1 * kernel_size2 * kernel_size2 * (((r2 + x1 * divisor) // kernel_size3) % kernel_size0) + 
                  (((r2 + x1 * divisor) % ((-2) + kernel_size2))))

        tmp3 = tl.load(input_ptr0 + index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(input_ptr1 + index0, rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)

        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        temp_result = temp_result + tmp8
        temp_result = tl.where(rmask & x_mask, temp_result, temp_result)

    result_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), result_sum, x_mask)