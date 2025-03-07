# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr, out_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x = (input_index % 21)
    input_y = input_index // 21
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_z = reduction_index

        temp_index_z = reduction_z + input_x * (triton_helpers.div_floor_integer(
            20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
            ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
            ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
            4 * kernel_size_x * kernel_size_y + 
            8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
            21
        ))

        temp_index_limit = ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                           ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                           ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                           4 * kernel_size_x * kernel_size_y + 
                           8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y

        index_within_limit = temp_index_z < temp_index_limit

        temp_value = tl.load(
            in_ptr + (((-1) * input_y) + 
                      ((-1) * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // ((-1) + 2 * kernel_size_y)) % ((-1) + 2 * kernel_size_y)))) + 
                      ((-16) * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x))) + 
                      ((-64) * kernel_size_y * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x))) + 
                      ((-4) * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // (1 + ((-4) * kernel_size_y) + 4 * kernel_size_y * kernel_size_y)) % ((-1) + 2 * kernel_size_x)))) + 
                      ((-4) * input_y * kernel_size_y * kernel_size_y) + 
                      2 * kernel_size_x * input_y + 
                      2 * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // ((-1) + 2 * kernel_size_y)) % ((-1) + 2 * kernel_size_y))) + 
                      4 * kernel_size_y * input_y + 
                      4 * kernel_size_y * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // (1 + ((-4) * kernel_size_y) + 4 * kernel_size_y * kernel_size_y)) % ((-1) + 2 * kernel_size_x))) + 
                      32 * kernel_size_x * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x)) + 
                      64 * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x)) + 
                      ((-128) * kernel_size_x * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x))) + 
                      ((-8) * kernel_size_x * kernel_size_y * input_y) + 
                      8 * kernel_size_x * input_y * kernel_size_y * kernel_size_y + 
                      128 * kernel_size_x * kernel_size_y * kernel_size_y * (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // kernel_size_z) % kernel_size_x)) + 
                      (((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      ))) % ((-1) + 2 * kernel_size_y))) + 
                      ((((reduction_z + input_x * (triton_helpers.div_floor_integer(
                          20 + ((-1) * kernel_size_x) + 2 * kernel_size_x * kernel_size_x + 
                          ((-8) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                          ((-4) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                          4 * kernel_size_x * kernel_size_y + 
                          8 * kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y, 
                          21
                      )) // (1 + ((-4) * kernel_size_y) + 4 * kernel_size_y * kernel_size_y)) % ((-1) + 2 * kernel_size_x)))), 
            reduction_mask & index_within_limit & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        broadcast_temp_value = tl.broadcast_to(temp_value, [XBLOCK, RBLOCK])
        temp_accumulator_update = temp_accumulator + broadcast_temp_value
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator_update, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr + (input_flat_index), temp_sum, input_mask)