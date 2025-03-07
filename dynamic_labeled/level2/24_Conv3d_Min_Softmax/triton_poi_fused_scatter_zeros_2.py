# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    mod_index = index % kernel_size1
    div_index2 = index // kernel_size2
    div_index1 = index // kernel_size1
    
    input0 = tl.load(in_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (mod_index + 4*div_index2 + div_index2*kernel_size3*kernel_size3 + ((-4)*kernel_size3*div_index2)), mask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (linear_index), mask, eviction_policy='evict_last')
    
    tl.device_assert(((0 <= input0) & (input0 < (-2) + kernel_size0)) | ~(mask), "index out of bounds: 0 <= input0 < (-2) + kernel_size0")
    
    neg_input1 = -input1
    mul_input3_input1 = input3 * input1
    fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_input1, input2, mul_input3_input1)
    
    store_index = (mod_index + 
                   ((-8)*div_index1) + 
                   4*input0 + 
                   input0*kernel_size3*kernel_size3 + 
                   ((-4)*kernel_size3*input0) + 
                   ((-2)*div_index1*kernel_size3*kernel_size3) + 
                   4*kernel_size0*div_index1 + 
                   8*kernel_size3*div_index1 + 
                   kernel_size0*div_index1*kernel_size3*kernel_size3 + 
                   ((-4)*kernel_size0*kernel_size3*div_index1))
    
    tl.store(out_ptr0 + (store_index), fused_multiply_add, mask)