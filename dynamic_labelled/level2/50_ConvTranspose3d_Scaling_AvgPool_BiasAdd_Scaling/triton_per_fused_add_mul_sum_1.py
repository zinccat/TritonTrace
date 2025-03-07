# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mul_sum_1per_fused_add_mul_sum_1(input_ptr, output_ptr, input_num_elements, output_num_elements):
    INPUT_BLOCK_SIZE: tl.constexpr = 1
    output_num_elements = 310
    OUTPUT_BLOCK_SIZE: tl.constexpr = 512
    input_offset = tl.program_id(0) * INPUT_BLOCK_SIZE
    tl.full([1], input_offset, tl.int32)
    tl.full([OUTPUT_BLOCK_SIZE], True, tl.int1)
    output_index = tl.arange(0, OUTPUT_BLOCK_SIZE)[:]
    output_mask = output_index < output_num_elements
    output_indices = output_index
    loaded_values = tl.load(input_ptr + (output_indices), output_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [OUTPUT_BLOCK_SIZE])
    masked_values = tl.where(output_mask, broadcasted_values, 0)
    summed_values = triton_helpers.promote_to_tensor(tl.sum(masked_values, 0))
    tl.store(output_ptr + (tl.full([1], 0, tl.int32)), summed_values, None)