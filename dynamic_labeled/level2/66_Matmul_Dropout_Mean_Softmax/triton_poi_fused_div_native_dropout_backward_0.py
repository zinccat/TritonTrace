# From: 66_Matmul_Dropout_Mean_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_native_dropout_backward_0(
    input_grad_ptr, dropout_mask_ptr, dropout_mask_bool_ptr, output_grad_ptr, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    batch_index = indices // 50
    element_index = indices

    input_grad = tl.load(input_grad_ptr + (batch_index), mask, eviction_policy='evict_last')
    dropout_mask = tl.load(dropout_mask_ptr + (batch_index), mask, eviction_policy='evict_last')
    dropout_mask_bool = tl.load(dropout_mask_bool_ptr + (element_index), mask).to(tl.int1)

    neg_input_grad = -input_grad
    scaled_input_grad = dropout_mask * input_grad
    fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_input_grad, scaled_input_grad, scaled_input_grad)

    dropout_scale = 0.02
    scaled_fma = fused_multiply_add * dropout_scale

    dropout_mask_float = dropout_mask_bool.to(tl.float32)
    dropout_rescale = 1.25
    rescaled_dropout_mask = dropout_mask_float * dropout_rescale

    final_output_grad = scaled_fma * rescaled_dropout_mask
    tl.store(output_grad_ptr + (element_index), final_output_grad, mask)