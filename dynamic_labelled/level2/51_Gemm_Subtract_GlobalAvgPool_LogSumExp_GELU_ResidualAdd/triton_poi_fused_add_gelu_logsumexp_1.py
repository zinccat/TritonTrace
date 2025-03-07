# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_logsumexp_1(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    block_index = index // 1024
    element_index = index

    input_value0 = tl.load(input_ptr0 + (block_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (element_index), mask)

    abs_value = tl.math.abs(input_value0)
    inf_value = float("inf")
    is_inf = abs_value == inf_value
    zero_value = 0.0
    max_value = tl.where(is_inf, zero_value, input_value0)

    adjusted_value = input_value0 - max_value
    exp_value = tl.math.exp(adjusted_value)
    log_value = tl.math.log(exp_value)
    logsumexp_value = log_value + max_value

    half = 0.5
    scaled_logsumexp = logsumexp_value * half

    erf_coefficient = 0.7071067811865476
    erf_input = logsumexp_value * erf_coefficient
    erf_value = tl.extra.cuda.libdevice.erf(erf_input)

    one_value = 1.0
    erf_adjusted = erf_value + one_value
    gelu_value = scaled_logsumexp * erf_adjusted

    fused_output = gelu_value + input_value1
    tl.store(output_ptr0 + (element_index), fused_output, mask)