# From: 34_InstanceNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(input_ptr, output_ptr, num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 1024
    reduction_num_elements = 65536
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = reduction_offset + r_base
        tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r_indices_flat = r_indices
        input_values = tl.load(input_ptr + (r_indices_flat + (65536 * x_indices_flat)), x_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_values, mean_accumulator, m2_accumulator, weight_accumulator, reduction_offset == 0
        )
        mean_accumulator = tl.where(x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(x_mask, weight_next, weight_accumulator)
    
    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_broadcasted = mean_final[:, None]
    variance_broadcasted = variance_final[:, None]
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = reduction_offset + r_base
        tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r_indices_flat = r_indices
        input_values = tl.load(input_ptr + (r_indices_flat + (65536 * x_indices_flat)), x_mask, eviction_policy='evict_first', other=0.0)
        centered_values = input_values - mean_broadcasted
        scale_factor = 65536.0
        variance_adjusted = variance_broadcasted / scale_factor
        epsilon = 1e-05
        variance_stabilized = variance_adjusted + epsilon
        normalized_values = centered_values * tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
        tl.store(output_ptr + (r_indices_flat + (65536 * x_indices_flat)), normalized_values, x_mask)