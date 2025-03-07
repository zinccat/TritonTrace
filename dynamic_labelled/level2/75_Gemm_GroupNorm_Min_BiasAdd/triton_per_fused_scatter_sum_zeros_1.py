# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_scatter_sum_zeros_1per_fused_scatter_sum_zeros_1(input_ptr0, input_ptr1, output_ptr1, kernel_size, x_num_elements, r_num_elements):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    
    r_index = tl.arange(0, RBLOCK)[:]
    
    input_value0 = tl.load(input_ptr0 + (x_index + kernel_size * r_index), None)
    input_value1 = tl.load(input_ptr1 + (x_index), None, eviction_policy='evict_last')
    
    broadcasted_value0 = tl.broadcast_to(input_value0, [RBLOCK])
    summed_value = triton_helpers.promote_to_tensor(tl.sum(broadcasted_value0, 0))
    
    tl.device_assert((0 <= input_value1) & (input_value1 < 256), "index out of bounds: 0 <= input_value1 < 256")
    
    tl.store(output_ptr1 + (input_value1 + 256 * x_index), summed_value, None)