# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_relu_66(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 12800
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x0 = (x_index % 1280)
    x1 = x_index // 1280
    x3 = x_index
    input_data = tl.load(in_ptr0 + (x0 + 1280 * r2 + 81920 * x1), x_mask, other=0.0)
    mean = tl.load(in_ptr1 + (x0), x_mask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x0), x_mask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), x_mask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), x_mask, eviction_policy='evict_last')
    
    centered_data = input_data - mean
    scaled_data = centered_data * variance
    normalized_data = scaled_data * gamma
    output_data = normalized_data + beta
    
    min_value = tl.full([1, 1], 0, tl.int32)
    relu_output = triton_helpers.maximum(min_value, output_data)
    broadcast_relu = tl.broadcast_to(relu_output, [XBLOCK, RBLOCK])
    masked_relu = tl.where(x_mask, broadcast_relu, 0)
    sum_relu = tl.sum(masked_relu, 1)[:, None]
    
    normalization_factor = 64.0
    mean_relu = sum_relu / normalization_factor
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), mean_relu, x_mask)