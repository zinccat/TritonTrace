# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_leaky_relu_leaky_relu_backward_mul_sum_0red_fused_leaky_relu_leaky_relu_backward_mul_sum_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    input_num_elements = 352
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_32 = x_index // 32
    x_mod_32 = x_index % 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_num_elements
        r_full_index = r_index
        temp_index = r_full_index + x_div_32 * ((10 + 2048 * kernel_size * kernel_size) // 11)
        max_index = 2048 * kernel_size * kernel_size
        index_mask = temp_index < max_index

        temp_load0 = tl.load(
            input_ptr0 + (128 * x_mod_32 * kernel_size * kernel_size + 
                          4096 * kernel_size * kernel_size * 
                          (((temp_index // (128 * kernel_size * kernel_size)) % 16) + 
                           (temp_index % (128 * kernel_size * kernel_size)))),
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        zero_value = 0.0
        positive_mask = temp_load0 > zero_value
        leaky_relu_slope = 0.2
        leaky_relu_output = tl.where(positive_mask, temp_load0, temp_load0 * leaky_relu_slope)

        temp_load1 = tl.load(input_ptr1 + (tl.broadcast_to(x_mod_32, [XBLOCK, RBLOCK])), r_mask & index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        elementwise_product = leaky_relu_output * temp_load1

        positive_mask2 = elementwise_product > zero_value
        temp_load2 = tl.load(
            input_ptr2 + (128 * x_mod_32 * kernel_size * kernel_size + 
                          4096 * kernel_size * kernel_size * 
                          (((temp_index // (128 * kernel_size * kernel_size)) % 16) + 
                           (temp_index % (128 * kernel_size * kernel_size)))),
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_load2_leaky = temp_load2 * leaky_relu_slope
        leaky_relu_output2 = tl.where(positive_mask2, temp_load2, temp_load2_leaky)
        final_product = leaky_relu_output2 * leaky_relu_output

        zero_filled = tl.full(final_product.shape, 0, final_product.dtype)
        masked_product = tl.where(index_mask, final_product, zero_filled)
        broadcasted_product = tl.broadcast_to(masked_product, [XBLOCK, RBLOCK])

        temp_accumulator = temp_accumulator + broadcasted_product
        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator, temp_accumulator)

    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x_full_index), reduced_sum, x_mask)