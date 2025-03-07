# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_9(input_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, kernel_size_c, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index // 16
    input_within_channel = input_index % 16
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_linear_index = reduction_index

        divisor = triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) +
            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w +
            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w +
            ((-4) * kernel_size_d * kernel_size_h * kernel_size_w), 21)

        temp_index = reduction_linear_index + input_channel * divisor
        temp_limit = ((-8) * kernel_size_d) + ((-2) * kernel_size_d * kernel_size_w * kernel_size_w) +
                     4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w +
                     kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w +
                     ((-4) * kernel_size_d * kernel_size_h * kernel_size_w)

        index_condition = temp_index < temp_limit

        input_offset_calculation = (
            ((-128) * (((temp_index // kernel_size_c) % kernel_size_d))) +
            ((-8) * input_within_channel) +
            ((-2) * (((temp_index // ((-2) + kernel_size_w)) % ((-2) + kernel_size_w)))) +
            4 * (((temp_index // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_size_h))) +
            kernel_size_w * (((temp_index // ((-2) + kernel_size_w)) % ((-2) + kernel_size_w))) +
            kernel_size_w * kernel_size_w * (((temp_index // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_size_h))) +
            ((-32) * kernel_size_w * kernel_size_w * (((temp_index // kernel_size_c) % kernel_size_d))) +
            ((-4) * kernel_size_w * (((temp_index // (4 + kernel_size_w * kernel_size_w + ((-4) * kernel_size_w))) % ((-2) + kernel_size_h)))) +
            ((-2) * input_within_channel * kernel_size_w * kernel_size_w) +
            4 * kernel_size_h * input_within_channel +
            8 * kernel_size_w * input_within_channel +
            64 * kernel_size_h * (((temp_index // kernel_size_c) % kernel_size_d)) +
            128 * kernel_size_w * (((temp_index // kernel_size_c) % kernel_size_d)) +
            kernel_size_h * input_within_channel * kernel_size_w * kernel_size_w +
            ((-64) * kernel_size_h * kernel_size_w * (((temp_index // kernel_size_c) % kernel_size_d))) +
            ((-4) * kernel_size_h * kernel_size_w * input_within_channel) +
            16 * kernel_size_h * kernel_size_w * kernel_size_w * (((temp_index // kernel_size_c) % kernel_size_d)) +
            (((temp_index % ((-2) + kernel_size_w)))))
        
        loaded_data = tl.load(input_ptr + input_offset_calculation, reduction_mask & index_condition & input_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
        temp_accumulator += broadcasted_data
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_linear_index), reduced_sum, input_mask)