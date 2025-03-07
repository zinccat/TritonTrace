# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_2(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    kernel_index0 = index % kernel_size0
    kernel_index1 = index // kernel_size1
    linear_index = index
    
    load_address0 = (kernel_index0 + 
                     ((-1) * kernel_index1) + 
                     ((-4) * kernel_index1 * kernel_size3 * kernel_size3) + 
                     2 * kernel_size2 * kernel_index1 + 
                     4 * kernel_size3 * kernel_index1 + 
                     ((-8) * kernel_size2 * kernel_size3 * kernel_index1) + 
                     8 * kernel_size2 * kernel_index1 * kernel_size3 * kernel_size3)
    
    tmp0 = tl.load(input_ptr0 + load_address0, mask, eviction_policy='evict_last')
    tmp2 = tl.load(input_ptr1 + load_address0, mask, eviction_policy='evict_last')
    
    tmp1 = -tmp0
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.extra.cuda.libdevice.tanh(tmp0)
    tmp6 = tmp5 * tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp4 * tmp8
    tmp10 = tmp9 * tmp0
    tmp11 = tl.extra.cuda.libdevice.fma(tmp1, tmp10, tmp10)
    tmp12 = 0.0625
    tmp13 = tmp11 * tmp12
    
    tl.store(output_ptr0 + linear_index, tmp13, mask)