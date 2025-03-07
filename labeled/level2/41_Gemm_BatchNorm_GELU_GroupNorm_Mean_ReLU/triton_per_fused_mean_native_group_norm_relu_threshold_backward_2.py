# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_mean_native_group_norm_relu_threshold_backward_2(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    # Load input tensors
    input_tensor0 = tl.load(in_ptr0 + (r1 + (1024 * x0)), None)
    input_tensor1 = tl.load(in_ptr1 + ((8 * x0) + (r1 // 128)), None, eviction_policy='evict_last')
    input_tensor2 = tl.load(in_ptr2 + ((8 * x0) + (r1 // 128)), None, eviction_policy='evict_last')
    input_tensor3 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    input_tensor4 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')

    # Constants
    half = 0.5
    sqrt2_over_2 = 0.7071067811865476
    one = 1.0
    one_over_128 = 128.0
    epsilon = 1e-05
    block_size = 1024.0
    zero = 0.0

    # Computation
    scaled_input = input_tensor0 * half
    erf_input = input_tensor0 * sqrt2_over_2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    scaled_erf = scaled_input * (erf_result + one)
    diff = scaled_erf - input_tensor1
    mean = input_tensor2 / one_over_128
    variance = mean + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance)
    normalized_diff = diff * inv_stddev
    weighted_diff = normalized_diff * input_tensor3
    output = weighted_diff + input_tensor4

    # Reduction
    output_broadcast = tl.broadcast_to(output, [RBLOCK])
    sum_output = triton_helpers.promote_to_tensor(tl.sum(output_broadcast, 0))
    mean_output = sum_output / block_size
    max_mean_output = triton_helpers.maximum(tl.full([1], 0, tl.int32), mean_output)

    # Thresholding
    threshold_condition = max_mean_output <= zero

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), max_mean_output, None)
    tl.store(out_ptr0 + (x0), threshold_condition, None)