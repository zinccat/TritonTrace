# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1per_fused__softmax_1(input_ptr, output_ptr_exp, output_ptr_sum, kernel_size_0, kernel_size_1, total_elements, reduced_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_mod_k0 = (x_indices % kernel_size_0)
    x_div_k0 = x_indices // kernel_size_0
    x_full_index = x_indices
    loaded_values = tl.load(input_ptr + (x_mod_k0 + 8192 * kernel_size_1 * r2 + 524288 * kernel_size_1 * x_div_k0), None, eviction_policy='evict_last')
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    max_values = triton_helpers.max2(broadcasted_values, 1)[:, None]
    shifted_values = loaded_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    sum_exp_values = tl.sum(broadcasted_exp_values, 1)[:, None]
    tl.store(output_ptr_exp + (x_full_index), max_values, None)
    tl.store(output_ptr_sum + (x_full_index), sum_exp_values, None)