# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_avg_pool2d_gelu_max_mul_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index

    input_val0 = tl.load(in_ptr0 + (4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_val1 = tl.load(in_ptr0 + (1 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_val2 = tl.load(in_ptr0 + (2 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_val3 = tl.load(in_ptr0 + (3 + 4 * r1 + 256 * x0), x_mask, eviction_policy='evict_last', other=0.0)

    sum1 = input_val1 + input_val0
    sum2 = input_val2 + sum1
    sum3 = input_val3 + sum2

    scale_factor = 0.25
    scaled_sum = sum3 * scale_factor

    gelu_factor = 0.5
    gelu_scaled = scaled_sum * gelu_factor

    erf_factor = 0.7071067811865476
    erf_input = scaled_sum * erf_factor

    erf_result = tl.extra.cuda.libdevice.erf(erf_input)

    erf_adjusted = erf_result + 1.0
    gelu_output = gelu_scaled * erf_adjusted

    final_output = gelu_output * 2.0
    broadcast_output = tl.broadcast_to(final_output, [XBLOCK, RBLOCK])

    masked_output = tl.where(x_mask, broadcast_output, float("-inf"))
    max_value, _ = triton_helpers.max2(masked_output, 1)[:, None]

    max_index, max_index_val = triton_helpers.max_with_index(masked_output, tl.broadcast_to(r_index, masked_output.shape), 1)

    tl.store(out_ptr0 + (r1 + 64 * x0), scaled_sum, x_mask)
    tl.store(out_ptr1 + (x0), max_value, x_mask)
    tl.store(out_ptr2 + (x0), max_index_val[:, None], x_mask)