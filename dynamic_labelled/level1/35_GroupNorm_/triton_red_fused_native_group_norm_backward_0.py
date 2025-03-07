# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_grad_ptr, kernel_size, input_num_elements, reduction_num_elements, 
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

        grad_input = tl.load(
            input_grad_ptr + (kernel_size * x_indices_flat * ((kernel_size * kernel_size) // kernel_size) + 
                              ((r_indices_flat % (kernel_size * ((kernel_size * kernel_size) // kernel_size))))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        input_data = tl.load(
            input_ptr + (kernel_size * (((r_indices_flat // ((kernel_size * kernel_size) // kernel_size)) % kernel_size)) + 
                         x_indices_flat * kernel_size * kernel_size + 
                         ((r_indices_flat % ((kernel_size * kernel_size) // kernel_size)))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        element_wise_product = grad_input * input_data
        broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_product
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)

    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_grad_ptr + (x_indices_flat), reduced_sum, x_mask)