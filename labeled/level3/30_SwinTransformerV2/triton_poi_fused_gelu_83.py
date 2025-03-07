# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_83poi_fused_gelu_83(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    global_indices = index_within_block
    input_values = tl.load(in_ptr0 + (global_indices), valid_indices_mask)
    half = 0.5
    scaled_input = input_values * half
    erf_coefficient = 0.7071067811865476
    erf_input = input_values * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = scaled_input * erf_adjusted
    tl.store(out_ptr0 + (global_indices), gelu_output, valid_indices_mask)