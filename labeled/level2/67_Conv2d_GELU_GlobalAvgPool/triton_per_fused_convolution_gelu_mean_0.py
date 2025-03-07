# From: 67_Conv2d_GELU_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_gelu_mean_0(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 900
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_mask = tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    row_indices = r_index
    col_indices = x_index
    col_offset = x_index % 16
    loaded_data = tl.load(in_out_ptr0 + (row_indices + (900 * col_indices)), r_mask, other=0.0)
    input_data = tl.load(in_ptr0 + (col_offset), None, eviction_policy='evict_last')
    accumulated_data = loaded_data + input_data
    half = 0.5
    scaled_data = accumulated_data * half
    erf_coefficient = 0.7071067811865476
    erf_input = accumulated_data * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_result = scaled_data * erf_adjusted
    broadcasted_result = tl.broadcast_to(gelu_result, [RBLOCK])
    masked_result = tl.where(r_mask, broadcasted_result, 0)
    sum_result = triton_helpers.promote_to_tensor(tl.sum(masked_result, 0))
    divisor = 900.0
    mean_result = sum_result / divisor
    tl.store(in_out_ptr0 + (row_indices + (900 * col_indices)), accumulated_data, r_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (col_indices), mean_result, None)