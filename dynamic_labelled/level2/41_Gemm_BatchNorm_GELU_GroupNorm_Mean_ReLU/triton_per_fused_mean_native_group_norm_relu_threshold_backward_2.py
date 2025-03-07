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

    # Load input tensors
    input_tensor0 = tl.load(in_ptr0 + (r1 + 1024 * x0), None)
    input_tensor1 = tl.load(in_ptr1 + (8 * x0 + (r1 // 128)), None, eviction_policy='evict_last')
    input_tensor2 = tl.load(in_ptr2 + (8 * x0 + (r1 // 128)), None, eviction_policy='evict_last')
    input_tensor3 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    input_tensor4 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')

    # Constants
    half = 0.5
    sqrt2_inv = 0.7071067811865476
    one = 1.0
    sqrt128_inv = 128.0
    epsilon = 1e-05
    block_size = 1024.0
    zero = 0.0

    # Computation
    scaled_input = input_tensor0 * half
    erf_input = input_tensor0 * sqrt2_inv
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    scaled_erf = scaled_input * (erf_result + one)
    diff = scaled_erf - input_tensor1
    mean = input_tensor2 / sqrt128_inv
    mean_plus_epsilon = mean + epsilon
    rsqrt = tl.extra.cuda.libdevice.rsqrt(mean_plus_epsilon)
    normalized_diff = diff * rsqrt
    weighted_diff = normalized_diff * input_tensor3
    result = weighted_diff + input_tensor4

    # Broadcast and reduce
    broadcast_result = tl.broadcast_to(result, [RBLOCK])
    sum_result = triton_helpers.promote_to_tensor(tl.sum(broadcast_result, 0))
    mean_result = sum_result / block_size
    max_result = triton_helpers.maximum(tl.full([1], 0, tl.int32), mean_result)
    threshold_condition = max_result <= zero

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), max_result, None)
    tl.store(out_ptr0 + (x0), threshold_condition, None)