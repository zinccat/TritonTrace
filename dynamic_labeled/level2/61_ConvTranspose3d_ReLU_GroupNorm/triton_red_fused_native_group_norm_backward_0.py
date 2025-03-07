# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices

        input_grad_index = (
            2 * (((r_indices_flat // (2 + kernel_size_0)) % (2 + kernel_size_0))) +
            4 * (r_indices_flat // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            8 * x_indices_flat +
            kernel_size_0 * (((r_indices_flat // (2 + kernel_size_0)) % (2 + kernel_size_0))) +
            kernel_size_0 * kernel_size_0 * (r_indices_flat // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            2 * x_indices_flat * kernel_size_0 * kernel_size_0 +
            4 * kernel_size_0 * (r_indices_flat // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) +
            4 * kernel_size_1 * x_indices_flat +
            8 * kernel_size_0 * x_indices_flat +
            kernel_size_1 * x_indices_flat * kernel_size_0 * kernel_size_0 +
            4 * kernel_size_0 * kernel_size_1 * x_indices_flat +
            ((r_indices_flat % (2 + kernel_size_0)))
        )

        input_grad_values = tl.load(input_grad_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_values = tl.load(input_ptr + input_grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        zero_tensor = tl.full([1, 1], 0, tl.int32)
        max_values = triton_helpers.maximum(zero_tensor, input_values)
        elementwise_product = input_grad_values * max_values
        broadcasted_product = tl.broadcast_to(elementwise_product, [XBLOCK, RBLOCK])
        temp_accumulator += broadcasted_product

        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator, temp_accumulator)

    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), reduced_sum, x_mask)