# From: 38_L1Norm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_abs_sum_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod2 = input_index % 2
    input_index_div2 = input_index // 2
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_full = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_full = reduction_index
        temp_index = reduction_index_full + input_index_mod2 * ((1 + kernel_size) // 2)
        kernel_limit = kernel_size
        within_kernel_limit = temp_index < kernel_limit
        loaded_value = tl.load(in_ptr0 + (reduction_index_full + kernel_size * input_index_div2 + input_index_mod2 * ((1 + kernel_size) // 2)), reduction_mask & within_kernel_limit & input_mask, eviction_policy='evict_first', other=0.0)
        abs_value = tl.math.abs(loaded_value)
        zero_filled = tl.full(abs_value.shape, 0, abs_value.dtype)
        conditional_abs = tl.where(within_kernel_limit, abs_value, zero_filled)
        broadcasted_conditional_abs = tl.broadcast_to(conditional_abs, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_conditional_abs
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum_update, temp_sum)

    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_index_full), reduced_sum, input_mask)