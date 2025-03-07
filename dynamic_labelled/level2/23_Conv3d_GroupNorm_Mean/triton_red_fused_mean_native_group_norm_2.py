# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_native_group_norm_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0,
    kernel_size0, kernel_size1, x_num_elements, r_num_elements,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_mod_4 = x_indices % 4
    x_div_4 = x_indices // 4
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < r_num_elements
        r_indices_flat = r_indices

        # Load data from input pointers
        temp0 = tl.load(
            input_ptr0 + (
                ((-128) * x_div_4) +
                ((-8) * (
                    (((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16)
                )) +
                ((-2) * (
                    (((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                )) +
                4 * (
                    (((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) %
                     ((-2) + kernel_size0))
                ) +
                kernel_size1 * (
                    (((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                ) +
                kernel_size1 * kernel_size1 * (
                    (((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) %
                     ((-2) + kernel_size0))
                ) +
                ((-32) * x_div_4 * kernel_size1 * kernel_size1) +
                ((-4) * kernel_size1 * (
                    (((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        (4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) %
                     ((-2) + kernel_size0))
                )) +
                ((-2) * kernel_size1 * kernel_size1 * (
                    (((((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16))
                )) +
                4 * kernel_size0 * (
                    (((((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16))
                )) +
                8 * kernel_size1 * (
                    (((((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16))
                )) +
                64 * kernel_size0 * x_div_4 +
                128 * kernel_size1 * x_div_4 +
                kernel_size0 * kernel_size1 * kernel_size1 * (
                    (((((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16))
                )) +
                ((-64) * kernel_size0 * kernel_size1 * x_div_4) +
                ((-4) * kernel_size0 * kernel_size1 * (
                    (((((((r_indices_flat + ((-32) * x_mod_4) +
                        ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                        16 * kernel_size0 * x_mod_4 +
                        32 * kernel_size1 * x_mod_4 +
                        ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                        4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                        ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                         4 * kernel_size0 + 8 * kernel_size1 +
                         kernel_size0 * kernel_size1 * kernel_size1 +
                         ((-4) * kernel_size0 * kernel_size1))) % 16)) % 16))
                )) +
                16 * kernel_size0 * x_div_4 * kernel_size1 * kernel_size1 +
                (r_indices_flat % ((-2) + kernel_size1))
            ),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp1 = tl.load(
            input_ptr1 + (
                8 * x_div_4 +
                (((((((r_indices_flat + ((-32) * x_mod_4) +
                    ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                    16 * kernel_size0 * x_mod_4 +
                    32 * kernel_size1 * x_mod_4 +
                    ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                    4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                    ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                     4 * kernel_size0 + 8 * kernel_size1 +
                     kernel_size0 * kernel_size1 * kernel_size1 +
                     ((-4) * kernel_size0 * kernel_size1))) % 16)) // 2) % 8)
            ),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp3 = tl.load(
            input_ptr2 + (
                8 * x_div_4 +
                (((((((r_indices_flat + ((-32) * x_mod_4) +
                    ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                    16 * kernel_size0 * x_mod_4 +
                    32 * kernel_size1 * x_mod_4 +
                    ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                    4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                    ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                     4 * kernel_size0 + 8 * kernel_size1 +
                     kernel_size0 * kernel_size1 * kernel_size1 +
                     ((-4) * kernel_size0 * kernel_size1))) % 16)) // 2) % 8)
            ),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp11 = tl.load(
            input_ptr3 + (
                (((r_indices_flat + ((-32) * x_mod_4) +
                    ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                    16 * kernel_size0 * x_mod_4 +
                    32 * kernel_size1 * x_mod_4 +
                    ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                    4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                    ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                     4 * kernel_size0 + 8 * kernel_size1 +
                     kernel_size0 * kernel_size1 * kernel_size1 +
                     ((-4) * kernel_size0 * kernel_size1))) % 16)
            ),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp13 = tl.load(
            input_ptr4 + (
                (((r_indices_flat + ((-32) * x_mod_4) +
                    ((-8) * x_mod_4 * kernel_size1 * kernel_size1) +
                    16 * kernel_size0 * x_mod_4 +
                    32 * kernel_size1 * x_mod_4 +
                    ((-16) * kernel_size0 * kernel_size1 * x_mod_4) +
                    4 * kernel_size0 * x_mod_4 * kernel_size1 * kernel_size1) //
                    ((-8) + ((-2) * kernel_size1 * kernel_size1) +
                     4 * kernel_size0 + 8 * kernel_size1 +
                     kernel_size0 * kernel_size1 * kernel_size1 +
                     ((-4) * kernel_size0 * kernel_size1))) % 16)
            ),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp2 = temp0 - temp1
        temp4 = (
            (tl.full([], 0.0, tl.float64) * ((tl.full([], 0.0, tl.float64) >=
                ((-16) + ((-4) * kernel_size1 * kernel_size1) +
                 8 * kernel_size0 + 16 * kernel_size1 +
                 ((-8) * kernel_size0 * kernel_size1) +
                 2 * kernel_size0 * kernel_size1 * kernel_size1))) +
             ((-16) + ((-4) * kernel_size1 * kernel_size1) +
              8 * kernel_size0 + 16 * kernel_size1 +
              ((-8) * kernel_size0 * kernel_size1) +
              2 * kernel_size0 * kernel_size1 * kernel_size1) *
             (((-16) + ((-4) * kernel_size1 * kernel_size1) +
               8 * kernel_size0 + 16 * kernel_size1 +
               ((-8) * kernel_size0 * kernel_size1) +
               2 * kernel_size0 * kernel_size1 * kernel_size1) >
              (tl.full([], 0.0, tl.float64))))
        )
        temp5 = temp4.to(tl.float32)
        temp6 = temp3 / temp5
        epsilon = 1e-05
        temp8 = temp6 + epsilon
        temp9 = tl.extra.cuda.libdevice.rsqrt(temp8)
        temp10 = temp2 * temp9
        temp12 = temp10 * temp11
        temp14 = temp12 + temp13
        temp15 = tl.broadcast_to(temp14, [XBLOCK, RBLOCK])
        temp17 = temp_sum + temp15
        temp_sum = tl.where(r_mask & x_mask, temp17, temp_sum)

    temp16 = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_flat), temp16, x_mask)