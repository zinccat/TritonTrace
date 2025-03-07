# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_backward_data_1poi_fused__softmax_backward_data_1(in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    mod_kernel_size0 = index % kernel_size0
    div_kernel_size1 = index // kernel_size1
    loaded_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    complex_index = (mod_kernel_size0 + 
                     ((-8) * div_kernel_size1) + 
                     ((-2) * div_kernel_size1 * kernel_size3 * kernel_size3) + 
                     4 * kernel_size2 * div_kernel_size1 + 
                     8 * kernel_size3 * div_kernel_size1 + 
                     kernel_size2 * div_kernel_size1 * kernel_size3 * kernel_size3 + 
                     ((-4) * kernel_size2 * kernel_size3 * div_kernel_size1))
    loaded_input0 = tl.load(in_ptr0 + complex_index, mask, eviction_policy='evict_last')
    loaded_input1 = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    negated_loaded_value = -loaded_value
    product = loaded_input1 * loaded_value
    fused_multiply_add = tl.extra.cuda.libdevice.fma(negated_loaded_value, loaded_input0, product)
    tl.store(in_out_ptr0 + (linear_index), fused_multiply_add, mask)