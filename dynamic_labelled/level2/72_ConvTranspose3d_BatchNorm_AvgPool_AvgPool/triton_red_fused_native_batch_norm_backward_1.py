# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_1red_fused_native_batch_norm_backward_1(
    input_grad_ptr, input_ptr, weight_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    x_num_elements = 3984
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = (x_index % 249)
    x1 = x_index // 249
    temp_grad_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index
    temp_weight_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        temp_index = r2 + x0 * (triton_helpers.div_floor_integer(
            248 + ((-1) * kernel_size0) + ((-12) * kernel_size0 * kernel_size1 * kernel_size1) + 6 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size1 * kernel_size1 * kernel_size1, 
            249
        ))
        temp_index_limit = ((-1) * kernel_size0) + ((-12) * kernel_size0 * kernel_size1 * kernel_size1) + 6 * kernel_size0 * kernel_size1 + 8 * kernel_size0 * kernel_size1 * kernel_size1 * kernel_size1
        index_within_bounds = temp_index < temp_index_limit

        input_grad = tl.load(
            input_grad_ptr + (((-1) * x1) + ((-1) * (((temp_index // kernel_size2) % kernel_size2))) + 
            ((-16) * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0))) + 
            ((-192) * kernel_size1 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0))) + 
            ((-12) * x1 * kernel_size1 * kernel_size1) + ((-4) * kernel_size1 * (((temp_index // kernel_size3) % kernel_size2))) + 
            2 * kernel_size1 * (((temp_index // kernel_size2) % kernel_size2)) + 
            4 * kernel_size1 * kernel_size1 * (((temp_index // kernel_size3) % kernel_size2)) + 
            6 * kernel_size1 * x1 + 8 * x1 * kernel_size1 * kernel_size1 * kernel_size1 + 
            96 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            128 * kernel_size1 * kernel_size1 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            ((temp_index % kernel_size2)) + (((temp_index // kernel_size3) % kernel_size2))),
            r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0
        )

        broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        temp_grad_sum += broadcast_input_grad
        temp_grad_sum = tl.where(r_mask & x_mask, temp_grad_sum, temp_grad_sum)

        input_data = tl.load(
            input_ptr + (((-1) * x1) + ((-1) * (((temp_index // kernel_size2) % kernel_size2))) + 
            ((-16) * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0))) + 
            ((-192) * kernel_size1 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0))) + 
            ((-12) * x1 * kernel_size1 * kernel_size1) + ((-4) * kernel_size1 * (((temp_index // kernel_size3) % kernel_size2))) + 
            2 * kernel_size1 * (((temp_index // kernel_size2) % kernel_size2)) + 
            4 * kernel_size1 * kernel_size1 * (((temp_index // kernel_size3) % kernel_size2)) + 
            6 * kernel_size1 * x1 + 8 * x1 * kernel_size1 * kernel_size1 * kernel_size1 + 
            96 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            128 * kernel_size1 * kernel_size1 * kernel_size1 * (((temp_index // ((-1) + ((-12) * kernel_size1 * kernel_size1) + 6 * kernel_size1 + 8 * kernel_size1 * kernel_size1 * kernel_size1)) % kernel_size0)) + 
            ((temp_index % kernel_size2)) + (((temp_index // kernel_size3) % kernel_size2))),
            r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0
        )

        weight_data = tl.load(
            weight_ptr + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), 
            r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0
        )

        input_grad_diff = input_data - weight_data
        weight_grad = input_grad * input_grad_diff
        weight_grad_masked = tl.where(index_within_bounds, weight_grad, tl.full(weight_grad.shape, 0, weight_grad.dtype))
        broadcast_weight_grad = tl.broadcast_to(weight_grad_masked, [XBLOCK, RBLOCK])
        temp_weight_sum += broadcast_weight_grad
        temp_weight_sum = tl.where(r_mask & x_mask, temp_weight_sum, temp_weight_sum)

    output_grad0 = tl.sum(temp_grad_sum, 1)[:, None]
    output_grad1 = tl.sum(temp_weight_sum, 1)[:, None]
    tl.store(output_grad_ptr0 + (x3), output_grad0, x_mask)
    tl.store(output_grad_ptr1 + (x3), output_grad1, x_mask)