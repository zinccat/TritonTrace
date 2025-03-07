# From: 66_Matmul_Dropout_Mean_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_native_dropout_backward_0poi_fused_div_native_dropout_backward_0(
    input_grad_ptr, dropout_mask_ptr, dropout_mask_bool_ptr, output_grad_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    # Calculate indices for accessing input gradients and dropout masks
    input_grad_index = block_indices // 50
    dropout_mask_index = block_indices

    # Load input gradients and dropout masks
    input_grad = tl.load(input_grad_ptr + (input_grad_index), valid_mask, eviction_policy='evict_last')
    dropout_mask = tl.load(dropout_mask_ptr + (input_grad_index), valid_mask, eviction_policy='evict_last')
    dropout_mask_bool = tl.load(dropout_mask_bool_ptr + (dropout_mask_index), valid_mask).to(tl.int1)

    # Compute gradients
    neg_input_grad = -input_grad
    dropout_grad = dropout_mask * input_grad
    fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_input_grad, dropout_grad, dropout_grad)
    scale_factor = 0.02
    scaled_fused_grad = fused_multiply_add * scale_factor
    dropout_mask_float = dropout_mask_bool.to(tl.float32)
    dropout_scale = 1.25
    dropout_scaled_mask = dropout_mask_float * dropout_scale
    final_grad = scaled_fused_grad * dropout_scaled_mask

    # Store the result
    tl.store(output_grad_ptr + (dropout_mask_index), final_grad, valid_mask)