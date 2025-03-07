# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_2(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    # Load data
    input_data = tl.load(in_out_ptr0 + (xindex), xmask, eviction_policy='evict_last')
    input_data0 = tl.load(in_ptr0 + (xindex), xmask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (rindex), None, eviction_policy='evict_last')
    
    # Compute log
    log_input_data = tl.math.log(input_data)
    
    # Compute absolute value
    abs_input_data0 = tl.math.abs(input_data0)
    
    # Handle infinities
    inf_value = float("inf")
    is_inf = abs_input_data0 == inf_value
    zero_value = 0.0
    adjusted_input_data0 = tl.where(is_inf, zero_value, input_data0)
    
    # Add log and adjusted input
    log_plus_adjusted = log_input_data + adjusted_input_data0
    
    # Sigmoid computation
    sigmoid_offset = 3.0
    sigmoid_input = log_plus_adjusted + sigmoid_offset
    sigmoid_output = tl.sigmoid(sigmoid_input)
    
    # Element-wise multiplication
    elementwise_product = log_plus_adjusted * sigmoid_output
    
    # Scale by constant
    scale_factor = 0.16666666666666666
    scaled_product = elementwise_product * scale_factor
    
    # Subtract input_data1
    subtracted_result = scaled_product - input_data1
    
    # Clamp result between -1 and 1
    clamped_result = triton_helpers.minimum(
        triton_helpers.maximum(subtracted_result, -1.0), 1.0
    )
    
    # Broadcast clamped result
    broadcast_clamped = tl.broadcast_to(clamped_result, [XBLOCK, RBLOCK])
    
    # Apply mask and find max
    masked_clamped = tl.where(xmask, broadcast_clamped, float("-inf"))
    max_values, max_indices = triton_helpers.max_with_index(masked_clamped, rindex, 1)
    
    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (xindex), log_plus_adjusted, xmask)
    tl.store(out_ptr0 + (xindex), max_values[:, None], xmask)
    tl.store(out_ptr1 + (xindex), max_indices[:, None], xmask)