# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_div_mul_native_batch_norm_backward_1(
    input_grad_ptr, scale_ptr, bias_ptr, output_grad_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 32
    input_index_0 = (input_index % 32)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_1 * (
            (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
             kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
             2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
             4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
             8 * kernel_size_0 * kernel_size_1) // 11
        )
        temp_index_1 = 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + \
                       kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                       2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                       4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + \
                       8 * kernel_size_0 * kernel_size_1
        temp_mask_2 = temp_index_0 < temp_index_1

        temp_load_0 = tl.load(
            input_grad_ptr + (input_index_0 + 32 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            )),
            reduction_mask & temp_mask_2 & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )

        temp_broadcast_ks2 = tl.broadcast_to(kernel_size_2, [XBLOCK, RBLOCK])
        temp_broadcast_ks2_float = temp_broadcast_ks2.to(tl.float32)
        temp_div = temp_load_0 / temp_broadcast_ks2_float

        temp_load_1 = tl.load(
            scale_ptr + (2 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // (2 + kernel_size_1)) % (2 + kernel_size_1)
            ) + 4 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)
            ) + 8 * input_index_0 + 256 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // (2 + kernel_size_1)) % (2 + kernel_size_1)
            ) + kernel_size_1 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)
            ) + 2 * input_index_0 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 * input_index_0 + 
            4 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)
            ) + 8 * kernel_size_1 * input_index_0 + 64 * kernel_size_1 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + 128 * kernel_size_0 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + 256 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + kernel_size_0 * input_index_0 * kernel_size_1 * kernel_size_1 + 
            4 * kernel_size_0 * kernel_size_1 * input_index_0 + 32 * kernel_size_0 * kernel_size_1 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + 128 * kernel_size_0 * kernel_size_1 * (
                ((reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) // kernel_size_2) % kernel_size_0
            ) + (
                (reduction_index_2 + input_index_1 * (
                    (10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1) // 11
                )) % (2 + kernel_size_1)
            ),
            reduction_mask & temp_mask_2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_scalar_8 = 2.0
        temp_mul = temp_load_1 * temp_scalar_8

        temp_load_2 = tl.load(
            bias_ptr + (tl.broadcast_to(input_index_0, [XBLOCK, RBLOCK])),
            reduction_mask & temp_mask_2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_sub = temp_mul - temp_load_2
        temp_mul_2 = temp_div * temp_sub
        temp_full = tl.full(temp_mul_2.shape, 0, temp_mul_2.dtype)
        temp_where = tl.where(temp_mask_2, temp_mul_2, temp_full)
        temp_broadcast = tl.broadcast_to(temp_where, [XBLOCK, RBLOCK])
        temp_accumulate = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulate, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_grad_ptr + (input_index_3), temp_sum, input_mask)