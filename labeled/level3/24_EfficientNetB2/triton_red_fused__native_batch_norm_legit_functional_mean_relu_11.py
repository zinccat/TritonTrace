# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_relu_11red_fused__native_batch_norm_legit_functional_mean_relu_11(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input, 
    output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 18816
    rnumel = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 96)
    x_row = x_index // 96

    mean_val = tl.load(input_ptr_mean + (x_col), x_mask, eviction_policy='evict_last')
    var_val = tl.load(input_ptr_var + (x_col), x_mask, eviction_policy='evict_last')
    beta_val = tl.load(input_ptr_beta + (x_col), x_mask, eviction_policy='evict_last')
    gamma_val = tl.load(input_ptr_gamma + (x_col), x_mask, eviction_policy='evict_last')

    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_element = r_index

        input_val = tl.load(input_ptr_input + (x_col + 96 * r_element + 12288 * x_row), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        normalized_val = input_val - mean_val
        scaled_val = normalized_val * var_val
        activated_val = scaled_val * gamma_val
        biased_val = activated_val + beta_val

        max_val = tl.full([1, 1], 0, tl.int32)
        relu_val = triton_helpers.maximum(max_val, biased_val)
        broadcast_relu = tl.broadcast_to(relu_val, [XBLOCK, RBLOCK])

        accumulated_result += broadcast_relu
        accumulated_result = tl.where(r_mask & x_mask, accumulated_result, accumulated_result)

    result_sum = tl.sum(accumulated_result, 1)[:, None]
    tl.store(output_ptr + (x_full_index), result_sum, x_mask)