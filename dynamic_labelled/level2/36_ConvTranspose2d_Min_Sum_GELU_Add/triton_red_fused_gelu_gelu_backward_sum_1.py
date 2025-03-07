# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_sum_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % kernel_size0)
    input_x1 = input_index // kernel_size0
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index
        temp_load = tl.load(in_ptr0 + (input_x0 + 2 * kernel_size1 * reduction_r2 + 4 * input_x1 * kernel_size1 * kernel_size1), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_accumulate, temp_sum)

    temp_reduce = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_x3), temp_reduce, input_mask)

    sqrt_half = 0.7071067811865476
    temp_scaled = temp_reduce * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(temp_scaled)
    one = 1.0
    erf_plus_one = erf_result + one
    half = 0.5
    temp_erf_half = erf_plus_one * half
    temp_square = temp_reduce * temp_reduce
    neg_half = -0.5
    temp_exp_arg = temp_square * neg_half
    exp_result = tl.math.exp(temp_exp_arg)
    sqrt_two_pi = 0.3989422804014327
    temp_exp_scaled = exp_result * sqrt_two_pi
    temp_final = temp_reduce * temp_exp_scaled
    temp_gelu = temp_erf_half + temp_final

    tl.store(out_ptr1 + (input_x3), temp_gelu, input_mask)