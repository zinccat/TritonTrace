# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    reduction_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod_64 = input_index % 64
    input_index_div_64 = input_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index
        temp_load = tl.load(
            in_ptr0 + (
                64 * (((reduction_index_flat + 64 * kernel_size0 * kernel_size1 * input_index_div_64) // 64) % (128 * kernel_size1))
                + 8192 * kernel_size1 * input_index_mod_64
                + 524288 * kernel_size1 * (((reduction_index_flat + 64 * kernel_size0 * kernel_size1 * input_index_div_64) // (8192 * kernel_size1)) % kernel_size0)
                + (reduction_index_flat % 64)
            ),
            reduction_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index_flat), temp_result, None)