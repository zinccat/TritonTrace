# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_2poi_fused_add_gelu_2(in_ptr0, in_ptr1, out_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x0 = (index % kernel_size0)
    x2 = index // kernel_size1
    x1 = ((index // kernel_size0) % 16)
    x3 = index
    input0 = tl.load(in_ptr0 + (x0 + 2 * kernel_size2 * x2), mask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (x1), mask, eviction_policy='evict_last')
    half = 0.5
    scaled_input0 = input0 * half
    sqrt2_over2 = 0.7071067811865476
    scaled_input0_sqrt2 = input0 * sqrt2_over2
    erf_result = tl.extra.cuda.libdevice.erf(scaled_input0_sqrt2)
    one = 1.0
    erf_plus_one = erf_result + one
    gelu_result = scaled_input0 * erf_plus_one
    fused_result = gelu_result + input1
    tl.store(out_ptr0 + (x3), fused_result, mask)