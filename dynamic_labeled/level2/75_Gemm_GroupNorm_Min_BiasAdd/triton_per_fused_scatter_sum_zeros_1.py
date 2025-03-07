# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_scatter_sum_zeros_1(input_ptr0, input_ptr1, output_ptr1, kernel_size0, x_num_elements, r_num_elements):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    
    r_index = tl.arange(0, RBLOCK)[:]
    
    x0 = x_index
    r1 = r_index
    
    tmp0 = tl.load(input_ptr0 + (x0 + kernel_size0 * r1), None)
    tmp4 = tl.load(input_ptr1 + (x0), None, eviction_policy='evict_last')
    
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    
    tl.device_assert((0 <= tmp4) & (tmp4 < 256), "index out of bounds: 0 <= tmp4 < 256")
    
    tl.store(output_ptr1 + (tmp4 + 256 * x0), tmp3, None)