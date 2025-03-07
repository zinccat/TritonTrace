# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_native_batch_norm_backward_1red_fused_add_div_native_batch_norm_backward_1(
    input_grad0_ptr, input_grad1_ptr, input_grad2_ptr, input_grad3_ptr, 
    output_grad0_ptr, output_grad1_ptr, kernel_size, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 32
    x0 = (xindex % 32)
    temp_accumulator1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    temp_accumulator2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex

        # Calculate temporary indices and masks
        temp_index = r2 + 46 * x1 + x1 * (triton_helpers.div_floor_integer((-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11))
        temp_limit = 496 + ((-1984) * kernel_size + 1984 * kernel_size * kernel_size)
        index_mask = temp_index < temp_limit

        # Load input gradients
        input_grad0 = tl.load(
            input_grad0_ptr + (
                (-1) * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 31 * x0 
                + 992 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((-3968) * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))) 
                + ((-124) * kernel_size * x0) 
                + ((-4) * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))) 
                + 2 * kernel_size * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 4 * kernel_size * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31)) 
                + 124 * x0 * kernel_size * kernel_size 
                + 3968 * kernel_size * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((temp_index % ((-1) + 2 * kernel_size))) 
                + (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))
            ), 
            rmask & index_mask & xmask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_grad1 = tl.load(
            input_grad1_ptr + (x0 + 32 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))), 
            rmask & index_mask & xmask, 
            eviction_policy='evict_first', 
            other=0.0
        )

        # Calculate normalization factor
        normalization_factor = tl.broadcast_to(31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size, [XBLOCK, RBLOCK]).to(tl.float32)
        normalized_input_grad1 = input_grad1 / normalization_factor

        # Accumulate results
        accumulated_grad = input_grad0 + normalized_input_grad1
        accumulated_grad_masked = tl.where(index_mask, accumulated_grad, tl.full(accumulated_grad.shape, 0, accumulated_grad.dtype))
        broadcast_accumulated_grad = tl.broadcast_to(accumulated_grad_masked, [XBLOCK, RBLOCK])
        temp_accumulator1 += tl.where(rmask & xmask, broadcast_accumulated_grad, temp_accumulator1)

        # Load additional input gradients
        input_grad2 = tl.load(
            input_grad2_ptr + (
                (-1) * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 31 * x0 
                + 992 * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((-3968) * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16))) 
                + ((-124) * kernel_size * x0) 
                + ((-4) * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))) 
                + 2 * kernel_size * (((temp_index // ((-1) + 2 * kernel_size)) % ((-1) + 2 * kernel_size))) 
                + 4 * kernel_size * kernel_size * (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31)) 
                + 124 * x0 * kernel_size * kernel_size 
                + 3968 * kernel_size * kernel_size * (((temp_index // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)) % 16)) 
                + ((temp_index % ((-1) + 2 * kernel_size))) 
                + (((temp_index // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)) % 31))
            ), 
            rmask & index_mask & xmask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        input_grad3 = tl.load(
            input_grad3_ptr + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), 
            rmask & index_mask & xmask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        # Calculate and accumulate gradients
        diff_grad = accumulated_grad - input_grad3
        product_grad = accumulated_grad * diff_grad
        product_grad_masked = tl.where(index_mask, product_grad, tl.full(product_grad.shape, 0, product_grad.dtype))
        broadcast_product_grad = tl.broadcast_to(product_grad_masked, [XBLOCK, RBLOCK])
        temp_accumulator2 += tl.where(rmask & xmask, broadcast_product_grad, temp_accumulator2)

    # Sum and store results
    summed_temp_accumulator1 = tl.sum(temp_accumulator1, 1)[:, None]
    summed_temp_accumulator2 = tl.sum(temp_accumulator2, 1)[:, None]
    tl.store(output_grad0_ptr + (x3), summed_temp_accumulator1, xmask)
    tl.store(output_grad1_ptr + (x3), summed_temp_accumulator2, xmask)