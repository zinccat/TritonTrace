# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(input_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, kernel_size_c, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 3984
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index % 249
    input_depth = input_index // 249
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_depth = reduction_index

        temp_index = reduction_depth + input_channel * (triton_helpers.div_floor_integer(
            248 + (-1 * kernel_size_d) + (-12 * kernel_size_d * kernel_size_h * kernel_size_h) + 6 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_h * kernel_size_h * kernel_size_h, 
            249
        ))

        temp_limit = (-1 * kernel_size_d) + (-12 * kernel_size_d * kernel_size_h * kernel_size_h) + 6 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_h * kernel_size_h * kernel_size_h
        temp_condition = temp_index < temp_limit

        temp_load = tl.load(input_ptr + (
            (-1 * input_depth) + 
            (-1 * (((temp_index // kernel_size_w) % kernel_size_w))) + 
            (-16 * (((temp_index // ((-1) + (-12 * kernel_size_h * kernel_size_h) + 6 * kernel_size_h + 8 * kernel_size_h * kernel_size_h * kernel_size_h)) % kernel_size_d))) + 
            (-192 * kernel_size_h * kernel_size_h * (((temp_index // ((-1) + (-12 * kernel_size_h * kernel_size_h) + 6 * kernel_size_h + 8 * kernel_size_h * kernel_size_h * kernel_size_h)) % kernel_size_d))) + 
            (-12 * input_depth * kernel_size_h * kernel_size_h) + 
            (-4 * kernel_size_h * (((temp_index // kernel_size_c) % kernel_size_w))) + 
            2 * kernel_size_h * (((temp_index // kernel_size_w) % kernel_size_w)) + 
            4 * kernel_size_h * kernel_size_h * (((temp_index // kernel_size_c) % kernel_size_w)) + 
            6 * kernel_size_h * input_depth + 
            8 * input_depth * kernel_size_h * kernel_size_h * kernel_size_h + 
            96 * kernel_size_h * (((temp_index // ((-1) + (-12 * kernel_size_h * kernel_size_h) + 6 * kernel_size_h + 8 * kernel_size_h * kernel_size_h * kernel_size_h)) % kernel_size_d)) + 
            128 * kernel_size_h * kernel_size_h * kernel_size_h * (((temp_index // ((-1) + (-12 * kernel_size_h * kernel_size_h) + 6 * kernel_size_h + 8 * kernel_size_h * kernel_size_h * kernel_size_h)) % kernel_size_d)) + 
            ((temp_index % kernel_size_w)) + 
            (((temp_index // kernel_size_c) % kernel_size_w))
        ), reduction_mask & temp_condition & input_mask, eviction_policy='evict_last', other=0.0)

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_linear_index), temp_result, input_mask)