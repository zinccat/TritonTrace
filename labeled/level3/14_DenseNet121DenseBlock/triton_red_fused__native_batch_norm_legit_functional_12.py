# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_12red_fused__native_batch_norm_legit_functional_12(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 512
    reduction_num_elements = 125440
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 128)
    x_height = x_indices // 128
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_flat_index = r_indices
        input_data = tl.load(
            input_ptr + (224 * (((r_flat_index + reduction_num_elements * x_height) // 224) % 224) + 
                         50176 * x_channel + 
                         6422528 * ((r_flat_index + reduction_num_elements * x_height) // 50176) + 
                         (r_flat_index % 224)),
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_reduce(
            broadcast_input, tmp_mean, tmp_m2, tmp_weight, r_offset == 0
        )
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    final_mean, final_var, final_weight = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_weight = final_weight[:, None]

    tl.store(output_mean_ptr + (x_flat_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_index), final_var, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), final_weight, x_mask)