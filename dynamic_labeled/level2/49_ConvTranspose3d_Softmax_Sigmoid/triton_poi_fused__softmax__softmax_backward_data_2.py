# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_2(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, 
    xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x3 = x_index
    x0 = (x_index % kernel_size0)
    x2 = x_index // kernel_size1
    
    input_value = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    input_max = tl.load(in_ptr0 + (x0 + 8192 * kernel_size2 * x2), None, eviction_policy='evict_last')
    sum_exp = tl.load(in_ptr1 + (x0 + 8192 * kernel_size2 * x2), None, eviction_policy='evict_last')
    grad_output = tl.load(in_ptr2 + (x0 + 8192 * kernel_size2 * x2), None, eviction_policy='evict_last')
    output_value = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    
    shifted_input = input_value - input_max
    exp_input = tl.math.exp(shifted_input)
    softmax_output = exp_input / sum_exp
    neg_softmax_grad = -softmax_output
    
    updated_output = tl.extra.cuda.libdevice.fma(neg_softmax_grad, grad_output, output_value)
    tl.store(in_out_ptr0 + (x3), updated_output, None)