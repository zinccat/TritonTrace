# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_37(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 7840
    rnumel = 192
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 784)
    x_row = x_index // 784
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input0_value = tl.load(
            in_ptr0 + (r2 + 192 * (((x_col % 28) % 7)) + 1344 * (((x_col // 28) % 7)) + 9408 * (((x_col % 28) // 7)) + 37632 * (x_col // 196) + 150528 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        input1_value = tl.load(in_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        combined_value = input0_value + input1_value
        broadcasted_value = tl.broadcast_to(combined_value, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_value, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_broadcast = mean_final[:, None]
    variance_broadcast = variance_final[:, None]

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input0_value = tl.load(
            in_ptr0 + (r2 + 192 * (((x_col % 28) % 7)) + 1344 * (((x_col // 28) % 7)) + 9408 * (((x_col % 28) // 7)) + 37632 * (x_col // 196) + 150528 * x_row),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        input1_value = tl.load(in_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        in_out_value = tl.load(in_out_ptr0 + (r2 + 192 * x3), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input2_value = tl.load(in_ptr2 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        input3_value = tl.load(in_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        combined_value = input0_value + input1_value
        centered_value = combined_value - mean_broadcast
        epsilon = 1e-05
        normalized_value = centered_value * tl.extra.cuda.libdevice.rsqrt(variance_broadcast / 192.0 + epsilon)
        scaled_value = normalized_value * input2_value
        output_value = scaled_value + input3_value
        updated_in_out_value = in_out_value + output_value
        tl.store(out_ptr2 + (r2 + 192 * x3), normalized_value, r_mask & x_mask)
        tl.store(in_out_ptr0 + (r2 + 192 * x3), updated_in_out_value, r_mask & x_mask)

    scale_factor = tl.extra.cuda.libdevice.rsqrt(variance_final / 192.0 + 1e-05) * 0.005208333333333333
    tl.store(out_ptr3 + (x3), scale_factor, x_mask)