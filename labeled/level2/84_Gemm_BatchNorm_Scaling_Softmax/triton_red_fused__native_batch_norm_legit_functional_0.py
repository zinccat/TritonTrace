# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, 
    output_ptr_mean, output_ptr_var, output_ptr_scale, 
    num_elements, num_reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 512
    num_reduction_elements = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    batch_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr_mean + (x_indices_flat + (512 * r_indices_flat)), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        batch_mean_next, batch_m2_next, batch_weight_next = triton_helpers.welford_reduce(
            broadcast_input, batch_mean, batch_m2, batch_weight, r_offset == 0
        )
        batch_mean = tl.where(r_mask & x_mask, batch_mean_next, batch_mean)
        batch_m2 = tl.where(r_mask & x_mask, batch_m2_next, batch_m2)
        batch_weight = tl.where(r_mask & x_mask, batch_weight_next, batch_weight)

    batch_mean_final, batch_var_final, batch_weight_final = triton_helpers.welford(
        batch_mean, batch_m2, batch_weight, 1
    )
    batch_mean_final = batch_mean_final[:, None]
    batch_var_final = batch_var_final[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), batch_mean_final, x_mask)
    input_scale = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_scale + (x_indices_flat), x_mask, eviction_policy='evict_last')

    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(batch_var_final + epsilon)
    scale_factor = 1.0078740157480315
    adjusted_var = batch_var_final * scale_factor
    scale = 0.1
    adjusted_scale = adjusted_var * scale
    momentum = 0.9
    adjusted_input_scale = input_scale * momentum
    adjusted_input_shift = input_shift * momentum

    final_scale = adjusted_scale + adjusted_input_scale
    final_shift = batch_mean_final * scale + adjusted_input_shift

    tl.store(output_ptr_var + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_scale + (x_indices_flat), final_scale, x_mask)
    tl.store(output_ptr_scale + (x_indices_flat), final_shift, x_mask)