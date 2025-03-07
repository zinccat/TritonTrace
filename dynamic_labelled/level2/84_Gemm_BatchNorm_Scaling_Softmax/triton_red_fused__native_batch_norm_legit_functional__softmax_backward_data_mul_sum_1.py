# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__softmax_backward_data_mul_sum_1red_fused__native_batch_norm_legit_functional__softmax_backward_data_mul_sum_1(
    input_ptr, output_ptr, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < reduction_elements
        r_indices = r_index
        loaded_values = tl.load(input_ptr + (r_indices), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask, temp_sum, temp_accumulator)

    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), reduced_sum, None)