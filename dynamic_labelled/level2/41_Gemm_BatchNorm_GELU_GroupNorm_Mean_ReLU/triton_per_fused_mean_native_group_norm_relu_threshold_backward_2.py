# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

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

    # Load input data
    input_data = tl.load(in_ptr0 + (r1 + 1024 * x0), None)
    batch_norm_mean = tl.load(in_ptr1 + (8 * x0 + (r1 // 128)), None, eviction_policy='evict_last')
    batch_norm_var = tl.load(in_ptr2 + (8 * x0 + (r1 // 128)), None, eviction_policy='evict_last')
    group_norm_weight = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    group_norm_bias = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')

    # Compute GELU
    half = 0.5
    sqrt_2_over_sqrt_pi = 0.7071067811865476
    gelu_input = input_data * half
    erf_input = input_data * sqrt_2_over_sqrt_pi
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_result = gelu_input * (erf_result + 1.0)

    # Compute BatchNorm
    batch_norm_delta = gelu_result - batch_norm_mean
    batch_norm_scale = batch_norm_var / 128.0
    epsilon = 1e-05
    batch_norm_denom = batch_norm_scale + epsilon
    batch_norm_inv_denom = tl.extra.cuda.libdevice.rsqrt(batch_norm_denom)
    batch_norm_output = batch_norm_delta * batch_norm_inv_denom

    # Compute GroupNorm
    group_norm_output = batch_norm_output * group_norm_weight + group_norm_bias

    # Compute mean across RBLOCK
    group_norm_output_broadcast = tl.broadcast_to(group_norm_output, [RBLOCK])
    group_norm_output_sum = triton_helpers.promote_to_tensor(tl.sum(group_norm_output_broadcast, 0))
    mean_value = group_norm_output_sum / 1024.0

    # Thresholding
    zero = tl.full([1], 0, tl.int32)
    max_mean_value = triton_helpers.maximum(zero, mean_value)
    threshold = 0.0
    is_below_threshold = max_mean_value <= threshold

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), max_mean_value, None)
    tl.store(out_ptr0 + (x0), is_below_threshold, None)