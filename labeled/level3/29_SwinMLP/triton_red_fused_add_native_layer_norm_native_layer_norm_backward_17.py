# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_17(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr2, output_ptr3, output_ptr4, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 7840
    rnumel = 192
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 784)
    x1 = x_index // 784

    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index

        input_val0 = tl.load(input_ptr0 + (r2 + 192 * x3), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val1 = tl.load(input_ptr1 + (32 * (((x0 % 28) % 7)) + 224 * (((x0 // 28) % 7)) + 1568 * (r2 // 32) + 9408 * (((x0 % 28) // 7)) + 37632 * (x0 // 196) + 150528 * x1 + (r2 % 32)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val2 = tl.load(input_ptr2 + (7 * (((x0 // 28) % 7)) + 49 * (r2 // 32) + (((x0 % 28) % 7))), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        combined_input1 = input_val1 + input_val2
        combined_input0 = input_val0 + combined_input1

        broadcasted_input = tl.broadcast_to(combined_input0, [XBLOCK, RBLOCK])

        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_input, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
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

        input_val0 = tl.load(input_ptr0 + (r2 + 192 * x3), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_val1 = tl.load(input_ptr1 + (32 * (((x0 % 28) % 7)) + 224 * (((x0 // 28) % 7)) + 1568 * (r2 // 32) + 9408 * (((x0 % 28) // 7)) + 37632 * (x0 // 196) + 150528 * x1 + (r2 % 32)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_val2 = tl.load(input_ptr2 + (7 * (((x0 // 28) % 7)) + 49 * (r2 // 32) + (((x0 % 28) % 7))), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        input_val3 = tl.load(input_ptr3 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        input_val4 = tl.load(input_ptr4 + (r2), r_mask, eviction_policy='evict_last', other=0.0)

        combined_input1 = input_val1 + input_val2
        combined_input0 = input_val0 + combined_input1

        normalized_input = combined_input0 - mean_broadcast
        epsilon = 1e-05
        variance_adjusted = variance_broadcast / 192.0 + epsilon
        inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

        normalized_output = normalized_input * inv_stddev
        scaled_output = normalized_output * input_val3
        final_output = scaled_output + input_val4

        tl.store(output_ptr2 + (r2 + 192 * x3), normalized_output, r_mask & x_mask)
        tl.store(output_ptr3 + (r2 + 192 * x3), final_output, r_mask & x_mask)

    epsilon = 1e-05
    variance_adjusted = variance_final / 192.0 + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    scaling_factor = inv_stddev * 0.005208333333333333

    tl.store(output_ptr4 + (x3), scaling_factor, x_mask)