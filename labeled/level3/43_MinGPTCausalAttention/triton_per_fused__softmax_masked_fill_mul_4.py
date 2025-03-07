# From: 43_MinGPTCausalAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_masked_fill_mul_4per_fused__softmax_masked_fill_mul_4(in_out_ptr0, in_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    # Calculate offsets and indices
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]
    
    # Load input data
    mask = tl.load(in_ptr0 + (r_index + 512 * (x_index % 512)), None, eviction_policy='evict_last').to(tl.int1)
    input_data = tl.load(in_out_ptr0 + (r_index + 512 * x_index), None)
    
    # Constants
    scale_factor = 0.10206207261596577
    negative_inf = float("-inf")
    
    # Apply mask and scale
    scaled_data = input_data * scale_factor
    masked_data = tl.where(mask, negative_inf, scaled_data)
    
    # Compute softmax
    max_value = triton_helpers.promote_to_tensor(tl.max(masked_data, 0))
    shifted_data = masked_data - max_value
    exp_data = tl.math.exp(shifted_data)
    sum_exp = triton_helpers.promote_to_tensor(tl.sum(exp_data, 0))
    softmax_result = exp_data / sum_exp
    
    # Store result
    tl.store(in_out_ptr0 + (r_index + 512 * x_index), softmax_result, None)