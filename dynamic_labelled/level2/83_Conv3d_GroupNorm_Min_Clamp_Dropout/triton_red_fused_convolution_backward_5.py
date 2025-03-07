# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(input_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, kernel_size_c, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_buffer = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
            21
        ))

        temp_index_1 = ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                       4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                       kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w)

        temp_index_2 = temp_index_0 < temp_index_1

        loaded_data = tl.load(input_ptr + (
            ((-128) * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d))) + 
            ((-8) * input_index_0) + 
            ((-2) * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // ((-2) + kernel_size_w)) % ((-2) + kernel_size_w)))) + 
            4 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_size_h))) + 
            kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // ((-2) + kernel_size_w)) % ((-2) + kernel_size_w))) + 
            kernel_size_w * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_h))) + 
            ((-32) * kernel_size_w * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d))) + 
            ((-4) * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_h)))) + 
            ((-2) * input_index_0 * kernel_size_w * kernel_size_w) + 
            4 * kernel_h * input_index_0 + 
            8 * kernel_size_w * input_index_0 + 
            64 * kernel_h * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d)) + 
            128 * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d)) + 
            kernel_h * input_index_0 * kernel_size_w * kernel_size_w + 
            ((-64) * kernel_h * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d))) + 
            ((-4) * kernel_h * kernel_size_w * input_index_0) + 
            16 * kernel_h * kernel_size_w * kernel_size_w * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) // kernel_size_c) % kernel_size_d)) + 
            (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )) % ((-2) + kernel_size_w)))), reduction_mask & temp_index_2 & input_mask, eviction_policy='evict_last', other=0.0)

        temp_buffer_broadcast = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
        temp_buffer_sum = temp_buffer + temp_buffer_broadcast
        temp_buffer = tl.where(reduction_mask & input_mask, temp_buffer_sum, temp_buffer)

    temp_result = tl.sum(temp_buffer, 1)[:, None]
    tl.store(output_ptr + (input_index_3), temp_result, input_mask)