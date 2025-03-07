# From: 4_Conv2d_Mish_Mish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mish_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    batch_index = (x_index // 900) % 16
    
    # Load data
    in_out_value = tl.load(in_out_ptr0 + (x_index), None)
    input_value = tl.load(in_ptr0 + (batch_index), None, eviction_policy='evict_last')
    
    # Perform addition
    sum_value = in_out_value + input_value
    
    # Mish activation function
    threshold = 20.0
    exp_sum = tl.math.exp(sum_value)
    log1p_exp_sum = tl.extra.cuda.libdevice.log1p(exp_sum)
    mish_value = tl.where(sum_value > threshold, sum_value, log1p_exp_sum)
    tanh_mish = tl.extra.cuda.libdevice.tanh(mish_value)
    mish_result = sum_value * tanh_mish
    
    # Second Mish activation
    exp_mish_result = tl.math.exp(mish_result)
    log1p_exp_mish_result = tl.extra.cuda.libdevice.log1p(exp_mish_result)
    second_mish_value = tl.where(mish_result > threshold, mish_result, log1p_exp_mish_result)
    tanh_second_mish = tl.extra.cuda.libdevice.tanh(second_mish_value)
    final_mish_result = mish_result * tanh_second_mish
    
    # Store results
    tl.store(in_out_ptr0 + (x_index), sum_value, None)
    tl.store(out_ptr0 + (x_index), final_mish_result, None)