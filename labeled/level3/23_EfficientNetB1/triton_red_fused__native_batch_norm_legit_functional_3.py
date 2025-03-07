# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_3red_fused__native_batch_norm_legit_functional_3(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 288
    num_elements_r = 125
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_32 = x_indices % 32
    x_div_32 = x_indices // 32
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_2d = r_indices
        tmp0 = tl.load(input_ptr0 + (x_mod_32 + 32 * r_indices_2d + 4000 * x_div_32), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(input_ptr1 + (x_mod_32 + 32 * r_indices_2d + 4000 * x_div_32), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(input_ptr2 + (x_mod_32 + 32 * r_indices_2d + 4000 * x_div_32), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcast_tmp0 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        broadcast_tmp1 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        broadcast_tmp2 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_combine(
            tmp_mean, tmp_m2, tmp_weight,
            broadcast_tmp0, broadcast_tmp1, broadcast_tmp2
        )
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    tmp_mean_final, tmp_m2_final, tmp_weight_final = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )
    tmp_mean_final = tmp_mean_final[:, None]
    tmp_m2_final = tmp_m2_final[:, None]
    tmp_weight_final = tmp_weight_final[:, None]
    tl.store(output_ptr0 + (x_full_indices), tmp_mean_final, x_mask)
    tl.store(output_ptr1 + (x_full_indices), tmp_m2_final, x_mask)
    tl.store(output_ptr2 + (x_full_indices), tmp_weight_final, x_mask)