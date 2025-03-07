# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_layer_norm_0(input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_reduction_elements = 199729
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_21 = (x_indices % 21)
    x_div_21 = x_indices // 21
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_indices_flat = r_indices
        tmp_index = r_indices_flat + 199729 * x_mod_21
        max_index = tl.full([1, 1], 4194304, tl.int32)
        index_mask = tmp_index < max_index
        input_values = tl.load(input_ptr + (4194304 * x_div_21 + ((r_indices_flat + 199729 * x_mod_21) % 4194304)), index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        zero_values = tl.full(input_values.shape, 0, input_values.dtype)
        mask_values = tl.where(index_mask, 0.0, zero_values)
        one_values = tl.full(1.0, 0, 1.0.dtype)
        broadcast_mask = tl.where(index_mask, 1.0, one_values)
        broadcast_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        broadcast_mask_values = tl.broadcast_to(mask_values, [XBLOCK, RBLOCK])
        broadcast_broadcast_mask = tl.broadcast_to(broadcast_mask, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_combine(
            mean_accumulator, m2_accumulator, weight_accumulator,
            broadcast_input, broadcast_mask_values, broadcast_broadcast_mask
        )
        
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]
    
    tl.store(output_mean_ptr + (x_indices_flat), mean_result, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), variance_result, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), weight_result, x_mask)