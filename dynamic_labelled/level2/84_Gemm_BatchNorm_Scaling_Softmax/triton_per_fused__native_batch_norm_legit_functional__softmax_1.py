# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    rblock_indices = rindex
    xblock_indices = xindex

    input_data = tl.load(in_ptr0 + (rblock_indices + 512 * xblock_indices), None)
    mean = tl.load(in_ptr1 + (rblock_indices), None, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (rblock_indices), None, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (rblock_indices), None, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (rblock_indices), None, eviction_policy='evict_last')
    epsilon = tl.load(in_ptr5 + (0))
    epsilon_broadcast = tl.broadcast_to(epsilon, [RBLOCK])

    normalized_data = input_data - mean
    scaled_data = normalized_data * variance
    scaled_data = scaled_data * gamma
    shifted_data = scaled_data + beta
    relu_mask = epsilon_broadcast >= 0.0
    relu_activation = tl.where(relu_mask, 1.0, -1.0)
    activated_data = shifted_data * relu_activation
    activated_data_broadcast = tl.broadcast_to(activated_data, [RBLOCK])

    max_value = triton_helpers.promote_to_tensor(tl.max(activated_data_broadcast, 0))
    shifted_activated_data = activated_data - max_value
    relu_scaled_data = relu_activation * epsilon_broadcast
    exponentiated_data = shifted_activated_data * relu_scaled_data
    exp_data = tl.math.exp(exponentiated_data)
    exp_data_broadcast = tl.broadcast_to(exp_data, [RBLOCK])

    sum_exp_data = triton_helpers.promote_to_tensor(tl.sum(exp_data_broadcast, 0))
    softmax_output = exp_data / sum_exp_data

    tl.store(in_out_ptr0 + (rblock_indices + 512 * xblock_indices), softmax_output, None)