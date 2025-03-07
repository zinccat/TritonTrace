# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_hardswish_relu_1(
    in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = (xindex % ks0)
    x4 = xindex // ks0
    x5 = xindex

    # Load input with specific indexing and masking
    input_value = tl.load(
        in_ptr0 + (
            x3
            + ((-128) * x4)
            + ((-8) * r2)
            + ((-32) * x4 * ks2 * ks2)
            + ((-2) * r2 * ks2 * ks2)
            + 4 * ks1 * r2
            + 8 * ks2 * r2
            + 64 * ks1 * x4
            + 128 * ks2 * x4
            + ks1 * r2 * ks2 * ks2
            + ((-64) * ks1 * ks2 * x4)
            + ((-4) * ks1 * ks2 * r2)
            + 16 * ks1 * x4 * ks2 * ks2
        ),
        xmask,
        eviction_policy="evict_last",
        other=0.0,
    )

    # HardSwish operation
    bias = 3.0
    shifted_input = input_value + bias
    relu_min = 0.0
    relu_max = 6.0
    relu_clamped = triton_helpers.minimum(triton_helpers.maximum(shifted_input, relu_min), relu_max)
    hardswish_output = input_value * relu_clamped * 0.16666666666666666

    # Softmax preparation
    max_value = tl.full([1, 1], 0, tl.int32)
    max_hardswish = triton_helpers.maximum(max_value, hardswish_output)
    broadcast_max = tl.broadcast_to(max_hardswish, [XBLOCK, RBLOCK])
    shifted_hardswish = tl.where(xmask, broadcast_max, float("-inf"))
    max_across_blocks = triton_helpers.max2(shifted_hardswish, 1)[:, None]
    exp_input = hardswish_output - max_across_blocks
    exp_values = tl.math.exp(exp_input)
    broadcast_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp = tl.where(xmask, broadcast_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]

    # Store results
    tl.store(out_ptr0 + (x5), max_across_blocks, xmask)
    tl.store(out_ptr1 + (x5), sum_exp, xmask)