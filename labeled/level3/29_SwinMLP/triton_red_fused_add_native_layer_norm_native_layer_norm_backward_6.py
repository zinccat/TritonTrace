# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6red_fused_add_native_layer_norm_native_layer_norm_backward_6(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 31360
    rnumel = 96
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = (x_index % 3136)
    x1 = x_index // 3136
    x3 = x_index
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input0 = tl.load(in_ptr0 + (x0 + 3136 * r2 + 301056 * x1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input1 = tl.load(in_ptr1 + (32 * (((x0 % 56) % 7)) + 224 * (((x0 // 56) % 7)) + 1568 * (r2 // 32) + 4704 * (((x0 % 56)) // 7) + 37632 * (x0 // 392) + 301056 * x1 + ((r2 % 32))), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input2 = tl.load(in_ptr2 + (7 * (((x0 // 56) % 7)) + 49 * (r2 // 32) + (((x0 % 56)) % 7)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        in_out_value = tl.load(in_out_ptr0 + (r2 + 96 * x3), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input3 = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        
        sum_inputs = input1 + input2
        sum_all_inputs = input0 + sum_inputs
        sum_with_in_out = in_out_value + input3
        total_sum = sum_all_inputs + sum_with_in_out
        broadcasted_sum = tl.broadcast_to(total_sum, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_sum, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)
        
        tl.store(in_out_ptr0 + (r2 + 96 * x3), total_sum, r_mask & x_mask)

    mean_final, m2_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_final = mean_final[:, None]
    m2_final = m2_final[:, None]
    weight_final = weight_final[:, None]

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        in_out_value = tl.load(in_out_ptr0 + (r2 + 96 * x3), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        deviation = in_out_value - mean_final
        scale_factor = 96.0
        variance = m2_final / scale_factor
        epsilon = 1e-05
        adjusted_variance = variance + epsilon
        inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
        normalized_value = deviation * inv_std_dev
        
        tl.store(out_ptr2 + (r2 + 96 * x3), normalized_value, r_mask & x_mask)

    variance = m2_final / 96.0
    adjusted_variance = variance + 1e-05
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scale_factor = 0.010416666666666666
    final_scale = inv_std_dev * scale_factor
    
    tl.store(out_ptr3 + (x3), final_scale, x_mask)