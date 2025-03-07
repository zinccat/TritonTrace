# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_2(in_ptr0, in_ptr1, out_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1
    kernel_index1 = (index // kernel_size0) % 16
    linear_index = index

    input_value0 = tl.load(in_ptr0 + (kernel_index0 + 2 * kernel_size2 * kernel_index2), mask, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (kernel_index1), mask, eviction_policy='evict_last')

    half = 0.5
    scaled_input0 = input_value0 * half

    sqrt_half = 0.7071067811865476
    scaled_input_sqrt = input_value0 * sqrt_half

    erf_result = tl.extra.cuda.libdevice.erf(scaled_input_sqrt)

    one = 1.0
    erf_plus_one = erf_result + one

    gelu_result = scaled_input0 * erf_plus_one

    fused_result = gelu_result + input_value1

    tl.store(out_ptr0 + (linear_index), fused_result, mask)