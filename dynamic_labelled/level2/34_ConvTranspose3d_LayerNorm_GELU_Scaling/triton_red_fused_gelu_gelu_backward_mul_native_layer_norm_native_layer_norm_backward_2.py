# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_2(
    input_ptr, output_ptr, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 64
    num_elements_r = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        loaded_values = tl.load(input_ptr + (x_indices_flat + 64 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)
    
    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), summed_values, x_mask)