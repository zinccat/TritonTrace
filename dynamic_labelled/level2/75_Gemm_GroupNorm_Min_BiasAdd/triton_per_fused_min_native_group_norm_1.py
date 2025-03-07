# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_min_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input0 = tl.load(in_ptr0 + (r1 + 256 * x0), None)
    input1 = tl.load(in_ptr1 + (8 * x0 + (r1 // 32)), None, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (8 * x0 + (r1 // 32)), None, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    input4 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    
    normalized_input = input0 - input1
    divisor = 32.0
    normalized_mean = input2 / divisor
    epsilon = 1e-05
    adjusted_mean = normalized_mean + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_mean)
    scaled_input = normalized_input * reciprocal_sqrt
    weighted_input = scaled_input * input3
    biased_output = weighted_input + input4
    
    broadcast_output = tl.broadcast_to(biased_output, [RBLOCK])
    min_value = triton_helpers.promote_to_tensor(triton_helpers.min2(broadcast_output, 0))
    r_index_broadcast = tl.broadcast_to(r_index, broadcast_output.shape)
    min_value_tensor, min_index_tensor = triton_helpers.min_with_index(broadcast_output, r_index_broadcast, 0)
    min_index = triton_helpers.promote_to_tensor(min_index_tensor)
    
    tl.store(out_ptr0 + (x0), min_value, None)
    tl.store(out_ptr1 + (x0), min_index, None)