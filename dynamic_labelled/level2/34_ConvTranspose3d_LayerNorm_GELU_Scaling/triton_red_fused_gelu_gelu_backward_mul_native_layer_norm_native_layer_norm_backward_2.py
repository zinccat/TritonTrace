# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_2red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_2(
    input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 64
    num_elements_r = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_indices_1 = r_indices
        temp_load = tl.load(input_ptr + (x_indices_0 + 64 * r_indices_1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)
    
    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), temp_result, x_mask)