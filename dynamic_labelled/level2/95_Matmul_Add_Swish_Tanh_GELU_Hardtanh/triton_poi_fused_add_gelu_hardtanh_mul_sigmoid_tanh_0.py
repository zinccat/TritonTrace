# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_hardtanh_mul_sigmoid_tanh_0poi_fused_add_gelu_hardtanh_mul_sigmoid_tanh_0(
    input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    input_data0 = tl.load(input_ptr0 + (global_indices), valid_mask)
    input_data1 = tl.load(input_ptr1 + (local_indices), valid_mask, eviction_policy='evict_last')

    add_result = input_data0 + input_data1
    sigmoid_result = tl.sigmoid(add_result)
    gelu_input = sigmoid_result * add_result
    tanh_result = tl.extra.cuda.libdevice.tanh(gelu_input)

    half = 0.5
    scaled_tanh = tanh_result * half

    sqrt_half = 0.7071067811865476
    scaled_sqrt_half_tanh = tanh_result * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(scaled_sqrt_half_tanh)

    one = 1.0
    erf_plus_one = erf_result + one

    gelu_result = scaled_tanh * erf_plus_one

    negative_one = -1.0
    hardtanh_result = triton_helpers.maximum(gelu_result, negative_one)
    clamped_result = triton_helpers.minimum(hardtanh_result, one)

    tl.store(output_ptr0 + (global_indices), clamped_result, valid_mask)