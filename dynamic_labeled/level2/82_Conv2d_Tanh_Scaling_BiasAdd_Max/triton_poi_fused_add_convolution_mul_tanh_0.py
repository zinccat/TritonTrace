# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_mul_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK : tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    element_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = element_indices < num_elements
    global_index = element_indices
    channel_index = ((element_indices // kernel_size) % 16)
    
    input_output_data = tl.load(in_out_ptr0 + (global_index), valid_mask, eviction_policy='evict_last')
    input_data_0 = tl.load(in_ptr0 + (channel_index), valid_mask, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (channel_index), valid_mask, eviction_policy='evict_last')
    
    intermediate_sum = input_output_data + input_data_0
    tanh_result = tl.extra.cuda.libdevice.tanh(intermediate_sum)
    scaled_tanh = tanh_result * 2.0
    final_result = scaled_tanh + input_data_1
    
    tl.store(in_out_ptr0 + (global_index), intermediate_sum, valid_mask)
    tl.store(out_ptr0 + (global_index), final_result, valid_mask)