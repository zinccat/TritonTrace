# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_convolution_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, kernel_size0, kernel_size1, kernel_size2, kernel_size3, kernel_size4, num_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = (index // kernel_size0) % 16
    kernel_index1 = index % kernel_size1
    depth_index = index // kernel_size2
    
    # Load data with eviction policy
    output_data = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_data0 = tl.load(in_ptr0 + (batch_index), mask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (kernel_index1 + ((-8)*depth_index) + ((-2)*depth_index*kernel_size4*kernel_size4) + 4*kernel_size3*depth_index + 8*kernel_size4*depth_index + kernel_size3*depth_index*kernel_size4*kernel_size4 + ((-4)*kernel_size3*kernel_size4*depth_index)), mask, eviction_policy='evict_last')
    input_data2 = tl.load(in_ptr2 + (kernel_index1 + ((-8)*depth_index) + ((-2)*depth_index*kernel_size4*kernel_size4) + 4*kernel_size3*depth_index + 8*kernel_size4*depth_index + kernel_size3*depth_index*kernel_size4*kernel_size4 + ((-4)*kernel_size3*kernel_size4*depth_index)), mask, eviction_policy='evict_last')
    
    # Perform computations
    sum_data = output_data + input_data0
    diff_data = sum_data - input_data1
    exp_data = tl.math.exp(diff_data)
    softmax_output = exp_data / input_data2
    
    # Store the result
    tl.store(in_out_ptr0 + (linear_index), softmax_output, mask)