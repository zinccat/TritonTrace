# From: 40_Matmul_Scaling_ResidualAdd

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = x_index
    x0 = x_index % 128
    input_value = tl.load(in_out_ptr0 + (x2), None)
    input_addend = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    sum_result = input_value + input_addend
    scaling_factor = 0.5
    scaled_result = sum_result * scaling_factor
    final_result = scaled_result + sum_result
    tl.store(in_out_ptr0 + (x2), final_result, None)