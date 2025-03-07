# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_mul_sum_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0,
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, x_num_elements, r_num_elements,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 336
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_16 = x_index // 16
    x_mod_16 = x_index % 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index

        divisor = triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) +
            4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 +
            kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 +
            ((-4) * kernel_size0 * kernel_size1 * kernel_size2), 21
        )

        tmp0 = r2 + x_div_16 * divisor
        tmp1 = ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) +
               4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 +
               kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 +
               ((-4) * kernel_size0 * kernel_size1 * kernel_size2)
        tmp2 = tmp0 < tmp1

        tmp3 = tl.load(
            input_ptr0 + (((-128) * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) +
                          ((-8) * x_mod_16) +
                          ((-2) * (((r2 + x_div_16 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) +
                          4 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)) +
                          kernel_size2 * (((r2 + x_div_16 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) +
                          kernel_size2 * kernel_size2 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)) +
                          ((-32) * kernel_size2 * kernel_size2 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) +
                          ((-4) * kernel_size2 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))) +
                          ((-2) * x_mod_16 * kernel_size2 * kernel_size2) +
                          4 * kernel_size1 * x_mod_16 + 8 * kernel_size2 * x_mod_16 +
                          64 * kernel_size1 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0) +
                          128 * kernel_size2 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0) +
                          kernel_size1 * x_mod_16 * kernel_size2 * kernel_size2 +
                          ((-64) * kernel_size1 * kernel_size2 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) +
                          ((-4) * kernel_size1 * kernel_size2 * x_mod_16) +
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0) +
                          (((r2 + x_div_16 * divisor) % ((-2) + kernel_size2)))), rmask & tmp2 & x_mask,
            eviction_policy='evict_last', other=0.0
        )

        tmp4 = tl.load(
            input_ptr1 + (((-128) * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0)) +
                          ((-8) * x_mod_16) +
                          ((-2) * (((r2 + x_div_16 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) +
                          4 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)) +
                          kernel_size2 * (((r2 + x_div_16 * divisor) // ((-2) + kernel_size2)) % ((-2) + kernel_size2)) +
                          kernel_size2 * kernel_size2 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1)) +
                          ((-32) * kernel_size2 * kernel_size2 * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0)) +
                          ((-4) * kernel_size2 * (((r2 + x_div_16 * divisor) // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % ((-2) + kernel_size1))) +
                          ((-2) * x_mod_16 * kernel_size2 * kernel_size2) +
                          4 * kernel_size1 * x_mod_16 + 8 * kernel_size2 * x_mod_16 +
                          64 * kernel_size1 * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0) +
                          128 * kernel_size2 * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0) +
                          kernel_size1 * x_mod_16 * kernel_size2 * kernel_size2 +
                          ((-64) * kernel_size1 * kernel_size2 * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0)) +
                          ((-4) * kernel_size1 * kernel_size2 * x_mod_16) +
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * (((x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)) // 16) % kernel_size0) +
                          (((r2 + x_div_16 * divisor) % ((-2) + kernel_size2)))), rmask & tmp2 & x_mask,
            eviction_policy='evict_last', other=0.0
        )

        tmp5 = tl.load(input_ptr2 + (tl.broadcast_to(x_mod_16, [XBLOCK, RBLOCK])), rmask & tmp2 & x_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 * tmp5

        tmp7 = tl.load(
            input_ptr3 + (x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)),
            rmask & tmp2 & x_mask, eviction_policy='evict_first', other=0.0
        )

        tmp8 = tmp6 - tmp7

        tmp9 = tl.load(
            input_ptr4 + (x_mod_16 + 16 * (((r2 + x_div_16 * divisor) // kernel_size3) % kernel_size0)),
            rmask & tmp2 & x_mask, eviction_policy='evict_first', other=0.0
        )

        tmp10 = tmp8 * tmp9

        tmp11 = -1.0
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = 1.0
        tmp14 = triton_helpers.minimum(tmp12, tmp13)

        tmp15 = tmp3 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])

        temp_result = temp_result + tmp18
        temp_result = tl.where(rmask & x_mask, temp_result, temp_result)

    result_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), result_sum, x_mask)