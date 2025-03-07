# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_logsumexp_1per_fused_logsumexp_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_dim = reduction_index
    input_mod_k0 = input_index % kernel_size0
    input_div_k0 = input_index // kernel_size0
    input_linear_index = input_index
    loaded_values = tl.load(
        in_ptr0 + (
            input_mod_k0 + 
            ((-1) * reduction_dim) + 
            ((-16) * input_div_k0) + 
            ((-64) * input_div_k0 * kernel_size2 * kernel_size2) + 
            ((-4) * reduction_dim * kernel_size2 * kernel_size2) + 
            2 * kernel_size1 * reduction_dim + 
            4 * kernel_size2 * reduction_dim + 
            32 * kernel_size1 * input_div_k0 + 
            64 * kernel_size2 * input_div_k0 + 
            ((-128) * kernel_size1 * kernel_size2 * input_div_k0) + 
            ((-8) * kernel_size1 * kernel_size2 * reduction_dim) + 
            8 * kernel_size1 * reduction_dim * kernel_size2 * kernel_size2 + 
            128 * kernel_size1 * input_div_k0 * kernel_size2 * kernel_size2
        ), 
        input_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(input_mask, broadcasted_values, float("-inf"))
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    zero_value = 0.0
    adjusted_max_values = tl.where(is_inf, zero_value, max_values)
    shifted_values = loaded_values - adjusted_max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(input_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (input_linear_index), max_values, input_mask)
    tl.store(out_ptr1 + (input_linear_index), sum_exp_values, input_mask)