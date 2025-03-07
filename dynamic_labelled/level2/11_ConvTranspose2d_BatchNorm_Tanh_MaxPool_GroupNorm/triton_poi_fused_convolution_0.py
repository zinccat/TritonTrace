# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    block_indices = indices
    batch_indices = (indices // 4096) % 64
    output_value = tl.load(in_out_ptr + (block_indices), None)
    input_value = tl.load(input_ptr + (batch_indices), None, eviction_policy='evict_last')
    result_value = output_value + input_value
    tl.store(in_out_ptr + (block_indices), result_value, None)