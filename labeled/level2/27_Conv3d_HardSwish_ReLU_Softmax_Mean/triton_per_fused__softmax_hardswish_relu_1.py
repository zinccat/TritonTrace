# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_hardswish_relu_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1612800
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 12600
    x1 = (xindex // 12600)
    
    # Load input data
    input_data = tl.load(in_ptr0 + (x0 + (12600 * r2) + (201600 * x1)), xmask, other=0.0)
    
    # HardSwish operation
    bias = 3.0
    shifted_input = input_data + bias
    lower_bound = 0.0
    upper_bound = 6.0
    clipped_input = triton_helpers.minimum(triton_helpers.maximum(shifted_input, lower_bound), upper_bound)
    hardswish_output = input_data * clipped_input * 0.16666666666666666
    
    # ReLU operation
    relu_output = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), hardswish_output)
    
    # Softmax operation
    broadcast_relu = tl.broadcast_to(relu_output, [XBLOCK, RBLOCK])
    masked_relu = tl.where(xmask, broadcast_relu, float("-inf"))
    max_values = triton_helpers.max2(masked_relu, 1)[:, None]
    shifted_logits = relu_output - max_values
    exp_logits = tl.math.exp(shifted_logits)
    broadcast_exp = tl.broadcast_to(exp_logits, [XBLOCK, RBLOCK])
    masked_exp = tl.where(xmask, broadcast_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    
    # Store results
    tl.store(out_ptr0 + (x0 + (12608 * x1)), max_values, xmask)
    tl.store(out_ptr1 + (x0 + (12608 * x1)), sum_exp, xmask)