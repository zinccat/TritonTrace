# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean, output_ptr_var,
    kernel_size, num_elements, reduced_num_elements, XBLOCK: tl.constexpr
):
    num_elements = 32
    reduced_num_elements = 11
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < reduced_num_elements
    r1 = r_indices
    x0 = x_indices

    mean = tl.load(input_ptr_mean + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    var = tl.load(input_ptr_var + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    count = tl.load(input_ptr_count + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    count_selected = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_accum, var_accum, count_accum = triton_helpers.welford(mean_selected, var_selected, count_selected, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    normalization_factor = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
    normalization_factor_float = normalization_factor.to(tl.float32)
    mean_normalized = var_accum_expanded / normalization_factor_float
    epsilon = 1e-05
    mean_normalized_eps = mean_normalized + epsilon
    inv_sqrt = tl.extra.cuda.libdevice.rsqrt(mean_normalized_eps)

    variance_correction = (((15872 + ((-63488) * kernel_size) + 63488 * kernel_size * kernel_size) / 32) /
                           ((tl.full([], -1.00000000000000, tl.float64)) + 
                            ((15872 + ((-63488) * kernel_size) + 63488 * kernel_size * kernel_size) / 32)))
    variance_correction_float = variance_correction.to(tl.float32)
    mean_scaled = mean_normalized * variance_correction_float
    momentum = 0.1
    mean_momentum = mean_scaled * momentum
    running_mean_momentum = running_mean * 0.9
    updated_running_mean = mean_momentum + running_mean_momentum

    variance_momentum = mean_scaled * momentum
    running_var_momentum = running_var * 0.9
    updated_running_var = variance_momentum + running_var_momentum

    tl.store(output_ptr_normalized + (x0), inv_sqrt, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)
    tl.store(output_ptr_var + (x0), var_accum_expanded, x_mask)