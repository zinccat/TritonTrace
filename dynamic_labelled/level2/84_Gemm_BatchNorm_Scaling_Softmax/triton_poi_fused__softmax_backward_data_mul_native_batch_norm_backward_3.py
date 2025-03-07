# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_backward_data_mul_native_batch_norm_backward_3poi_fused__softmax_backward_data_mul_native_batch_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ks0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    x0 = xindex % 512

    softmax_input = tl.load(in_ptr0 + (x2), xmask)
    batch_norm_mean = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    softmax_output = tl.load(in_ptr2 + (x2), xmask)
    batch_norm_scale = tl.load(in_ptr3 + (0))
    batch_norm_scale_broadcast = tl.broadcast_to(batch_norm_scale, [XBLOCK])
    in_out_grad = tl.load(in_out_ptr0 + (x2), xmask)
    batch_norm_var = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    batch_norm_grad = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    batch_norm_input = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    batch_norm_grad_input = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    batch_norm_grad_output = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')

    neg_softmax_input = -softmax_input
    softmax_grad = softmax_output * softmax_input
    batch_norm_grad_adjusted = tl.extra.cuda.libdevice.fma(neg_softmax_input, batch_norm_mean, softmax_grad)
    grad_scaled = batch_norm_grad_adjusted * batch_norm_scale_broadcast
    grad_diff = in_out_grad - batch_norm_var

    normalization_factor = tl.full([], 1.00000000000000, tl.float64) / ((512 * ks0) / 512)
    normalization_factor_float32 = normalization_factor.to(tl.float32)
    var_scaled = batch_norm_grad * normalization_factor_float32
    var_squared = batch_norm_input * batch_norm_input
    var_scaled_squared = var_scaled * var_squared
    grad_diff_scaled = grad_diff * var_scaled_squared
    grad_adjusted = grad_scaled - grad_diff_scaled

    grad_input_scaled = batch_norm_grad_input * normalization_factor_float32
    final_grad = grad_adjusted - grad_input_scaled
    grad_output_scaled = batch_norm_input * batch_norm_grad_output
    final_grad_adjusted = final_grad * grad_output_scaled

    tl.store(in_out_ptr0 + (x2), final_grad_adjusted, xmask)