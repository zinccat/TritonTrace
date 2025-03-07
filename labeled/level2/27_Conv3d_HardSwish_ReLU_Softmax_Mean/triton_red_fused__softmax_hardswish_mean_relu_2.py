# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__softmax_hardswish_mean_relu_2(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 12600
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x1 = (x_index // 16)
    softmax_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input0 = tl.load(in_ptr0 + (r2 + (12600 * x3)), r_mask, eviction_policy='evict_first', other=0.0)
        input1 = tl.load(in_ptr1 + (r2 + (12608 * x1)), r_mask, eviction_policy='evict_last', other=0.0)
        input2 = tl.load(in_ptr2 + (r2 + (12608 * x1)), r_mask, eviction_policy='evict_last', other=0.0)
        
        bias = 3.0
        biased_input = input0 + bias
        lower_bound = 0.0
        upper_bound = 6.0
        
        clipped_input = triton_helpers.minimum(triton_helpers.maximum(biased_input, lower_bound), upper_bound)
        scaled_input = clipped_input * input0
        scale_factor = 0.16666666666666666
        scaled_clipped_input = scaled_input * scale_factor
        
        max_value = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), scaled_clipped_input)
        shifted_input = max_value - input1
        exp_shifted = tl.math.exp(shifted_input)
        softmax_component = exp_shifted / input2
        broadcasted_component = tl.broadcast_to(softmax_component, [XBLOCK, RBLOCK])
        
        softmax_accumulator = tl.where(r_mask, softmax_accumulator + broadcasted_component, softmax_accumulator)
    
    softmax_sum = tl.sum(softmax_accumulator, 1)[:, None]
    normalization_factor = 12600.0
    mean_softmax = softmax_sum / normalization_factor
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), mean_softmax, None)