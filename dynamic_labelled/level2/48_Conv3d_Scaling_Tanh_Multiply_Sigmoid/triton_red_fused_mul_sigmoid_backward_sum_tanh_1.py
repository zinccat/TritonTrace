# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sigmoid_backward_sum_tanh_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 336
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_indices // 16
    x0 = x_indices % 16
    temp_buffer1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_indices
    temp_buffer2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r2 = r_indices
        index_offset = r2 + x1 * (
            triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + 
                kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + 
                ((-4) * kernel_size0 * kernel_size1 * kernel_size2), 
                21
            )
        )
        index_limit = ((-8) * kernel_size0) + ((-2) * kernel_size0 * kernel_size2 * kernel_size2) + \
                      4 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size2 + \
                      kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2 + \
                      ((-4) * kernel_size0 * kernel_size1 * kernel_size2)
        index_condition = index_offset < index_limit

        input_value0 = tl.load(
            input_ptr0 + (
                ((-128) * (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-8) * x0) + 
                ((-2) * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2)))) + 
                4 * (((index_offset // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % 
                      ((-2) + kernel_size1))) + 
                kernel_size2 * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                kernel_size2 * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                ((-32) * kernel_size2 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1)))) + 
                ((-2) * x0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * x0 + 8 * kernel_size2 * x0 + 
                64 * kernel_size1 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                128 * kernel_size2 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                ((-64) * kernel_size1 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size1 * kernel_size2 * x0) + 
                16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                (index_offset % ((-2) + kernel_size2))
            ), 
            r_mask & index_condition & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_value1 = tl.load(
            input_ptr1 + (
                ((-128) * (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-8) * x0) + 
                ((-2) * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2)))) + 
                4 * (((index_offset // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % 
                      ((-2) + kernel_size1))) + 
                kernel_size2 * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                kernel_size2 * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                ((-32) * kernel_size2 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1)))) + 
                ((-2) * x0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * x0 + 8 * kernel_size2 * x0 + 
                64 * kernel_size1 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                128 * kernel_size2 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                ((-64) * kernel_size1 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size1 * kernel_size2 * x0) + 
                16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                (index_offset % ((-2) + kernel_size2))
            ), 
            r_mask & index_condition & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        sigmoid_input = 1.0 - input_value1
        sigmoid_output = input_value1 * sigmoid_input
        masked_sigmoid_output = input_value0 * sigmoid_output

        input_value2 = tl.load(
            input_ptr2 + (
                ((-128) * (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-8) * x0) + 
                ((-2) * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2)))) + 
                4 * (((index_offset // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % 
                      ((-2) + kernel_size1))) + 
                kernel_size2 * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                kernel_size2 * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                ((-32) * kernel_size2 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1)))) + 
                ((-2) * x0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * x0 + 8 * kernel_size2 * x0 + 
                64 * kernel_size1 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                128 * kernel_size2 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                ((-64) * kernel_size1 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size1 * kernel_size2 * x0) + 
                16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                (index_offset % ((-2) + kernel_size2))
            ), 
            r_mask & index_condition & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_value3 = tl.load(
            input_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), 
            r_mask & index_condition & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        tanh_input = input_value2 * input_value3
        tanh_output = tl.extra.cuda.libdevice.tanh(tanh_input)
        masked_tanh_output = masked_sigmoid_output * tanh_output

        temp_result1 = tl.full(masked_tanh_output.shape, 0, masked_tanh_output.dtype)
        conditional_result1 = tl.where(index_condition, masked_tanh_output, temp_result1)
        broadcast_result1 = tl.broadcast_to(conditional_result1, [XBLOCK, RBLOCK])
        temp_buffer1 += tl.where(r_mask & x_mask, broadcast_result1, temp_buffer1)

        input_value4 = tl.load(
            input_ptr4 + (
                ((-128) * (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-8) * x0) + 
                ((-2) * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2)))) + 
                4 * (((index_offset // (4 + kernel_size2 * kernel_size2 + ((-4) * kernel_size2))) % 
                      ((-2) + kernel_size1))) + 
                kernel_size2 * (((index_offset // ((-2) + kernel_size2)) % ((-2) + kernel_size2))) + 
                kernel_size2 * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1))) + 
                ((-32) * kernel_size2 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size2 * (((index_offset // (4 + kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size2))) % ((-2) + kernel_size1)))) + 
                ((-2) * x0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * x0 + 8 * kernel_size2 * x0 + 
                64 * kernel_size1 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                128 * kernel_size2 * (((index_offset // 
                    ((-8) + ((-2) * kernel_size2 * kernel_size2) + 4 * kernel_size1 + 
                    8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                ((-64) * kernel_size1 * kernel_size2 * 
                 (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                     4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                     ((-4) * kernel_size1 * kernel_size2))) % kernel_size0))) + 
                ((-4) * kernel_size1 * kernel_size2 * x0) + 
                16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                (((index_offset // ((-8) + ((-2) * kernel_size2 * kernel_size2) + 
                    4 * kernel_size1 + 8 * kernel_size2 + kernel_size1 * kernel_size2 * kernel_size2 + 
                    ((-4) * kernel_size1 * kernel_size2))) % kernel_size0)) + 
                (index_offset % ((-2) + kernel_size2))
            ), 
            r_mask & index_condition & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        intermediate_result2 = input_value4 * input_value2
        temp_result2 = tl.full(intermediate_result2.shape, 0, intermediate_result2.dtype)
        conditional_result2 = tl.where(index_condition, intermediate_result2, temp_result2)
        broadcast_result2 = tl.broadcast_to(conditional_result2, [XBLOCK, RBLOCK])
        temp_buffer2 += tl.where(r_mask & x_mask, broadcast_result2, temp_buffer2)

    final_result1 = tl.sum(temp_buffer1, 1)[:, None]
    final_result2 = tl.sum(temp_buffer2, 1)[:, None]
    tl.store(output_ptr0 + (x3), final_result1, x_mask)
    tl.store(output_ptr1 + (x3), final_result2, x_mask)