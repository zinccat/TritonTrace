# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_1(
    input_grad_ptr, input_ptr, weight_ptr, output_grad_ptr0, output_grad_ptr1, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, input_num_elements, 
    reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 3984
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_249 = x_index % 249
    x_div_249 = x_index // 249
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_index_flat = x_index
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_num_elements
        r_index_flat = r_index

        temp_index0 = r_index_flat + x_mod_249 * (
            triton_helpers.div_floor_integer(
                248 + ((-1) * kernel_size0) + ((-12) * kernel_size0 * kernel_size1 * kernel_size1) + 
                6 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size1 * kernel_size1 * kernel_size1, 
                249
            )
        )
        temp_index_limit = ((-1) * kernel_size0) + ((-12) * kernel_size0 * kernel_size1 * kernel_size1) + \
                           6 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size1 * kernel_size1 * kernel_size1
        index_within_bounds = temp_index0 < temp_index_limit

        input_index = (
            (-1) * x_div_249 + 
            (-1) * (((temp_index0 // kernel_size2) % kernel_size2)) + 
            (-16) * (((temp_index0 // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            (-192) * kernel_size1 * kernel_size1 * (((temp_index0 // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            (-12) * x_div_249 * kernel_size1 * kernel_size1 + 
            (-4) * kernel_size1 * (((temp_index0 // kernel_size3) % kernel_size2)) + 
            2 * kernel_size1 * (((temp_index0 // kernel_size2) % kernel_size2)) + 
            4 * kernel_size1 * kernel_size1 * (((temp_index0 // kernel_size3) % kernel_size2)) + 
            6 * kernel_size1 * x_div_249 + 
            8 * x_div_249 * kernel_size1 * kernel_size1 * kernel_size1 + 
            96 * kernel_size1 * (((temp_index0 // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            128 * kernel_size1 * kernel_size1 * kernel_size1 * (((temp_index0 // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            ((temp_index0 % kernel_size2)) + (((temp_index0 // kernel_size3) % kernel_size2))
        )

        input_grad = tl.load(input_grad_ptr + input_index, r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        temp_sum0 = temp_sum0 + broadcast_input_grad
        temp_sum0 = tl.where(r_mask & x_mask, temp_sum0, temp_sum0)

        input = tl.load(input_ptr + input_index, r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        weight = tl.load(weight_ptr + tl.broadcast_to(x_div_249, [XBLOCK, RBLOCK]), r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        diff = input - weight
        product = input_grad * diff
        product_masked = tl.where(index_within_bounds, product, tl.full(product.shape, 0, product.dtype))
        broadcast_product = tl.broadcast_to(product_masked, [XBLOCK, RBLOCK])
        temp_sum1 = temp_sum1 + broadcast_product
        temp_sum1 = tl.where(r_mask & x_mask, temp_sum1, temp_sum1)

    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + x_index_flat, output_grad0, x_mask)
    tl.store(output_grad_ptr1 + x_index_flat, output_grad1, x_mask)