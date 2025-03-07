# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_7(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data
    input_data0 = tl.load(in_ptr0 + (r1 + 768 * x0), r_mask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (r1 + 768 * x0), r_mask, other=0.0)
    input_data2 = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    input_data3 = tl.load(in_ptr3 + (r1), r_mask, eviction_policy='evict_last', other=0.0)

    # Compute intermediate values
    sum_inputs = input_data0 + input_data1
    broadcast_sum = tl.broadcast_to(sum_inputs, [RBLOCK])
    masked_broadcast = tl.where(r_mask, broadcast_sum, 0)

    # Compute mean
    sum_masked = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast, 0))
    num_elements = tl.full([1], 768, tl.int32).to(tl.float32)
    mean = sum_masked / num_elements

    # Compute variance
    centered_data = broadcast_sum - mean
    squared_data = centered_data * centered_data
    broadcast_squared = tl.broadcast_to(squared_data, [RBLOCK])
    masked_squared = tl.where(r_mask, broadcast_squared, 0)
    sum_squared = triton_helpers.promote_to_tensor(tl.sum(masked_squared, 0))
    variance = sum_squared / 768.0

    # Compute layer normalization
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance + epsilon)
    normalized_data = (sum_inputs - mean) * inv_std
    output_data = normalized_data * input_data2 + input_data3

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std, None)
    tl.store(out_ptr1 + (r1 + 768 * x0), output_data, r_mask)
    tl.store(out_ptr0 + (x0), mean, None)