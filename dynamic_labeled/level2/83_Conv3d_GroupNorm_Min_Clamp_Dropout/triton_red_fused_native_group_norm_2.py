# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_2(input_output_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r1 = r_index
        load_index = r1 + ((-16)*x0) + ((-4)*x0*kernel_size_1*kernel_size_1) + 8*kernel_size_0*x0 + 16*kernel_size_1*x0 + ((-8)*kernel_size_0*kernel_size_1*x0) + 2*kernel_size_0*x0*kernel_size_1*kernel_size_1
        tmp0 = tl.load(input_ptr + load_index, r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            tmp1, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]
    
    tl.store(output_ptr + (x0), mean, x_mask)
    
    clamp_min = tl.full([], 0.0, tl.float64)
    clamp_max = tl.full([], 0.0, tl.float64)
    index_expr = ((-16) + ((-4)*kernel_size_1*kernel_size_1) + 8*kernel_size_0 + 16*kernel_size_1 + ((-8)*kernel_size_0*kernel_size_1) + 2*kernel_size_0*kernel_size_1*kernel_size_1)
    clamped_index = tl.where(index_expr >= clamp_min, index_expr, clamp_min)
    clamped_index = tl.where(index_expr <= clamp_max, clamped_index, clamp_max)
    clamped_index = clamped_index.to(tl.float32)
    
    normalized_variance = variance / clamped_index
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    tl.debug_barrier()
    tl.store(input_output_ptr + (x0), inv_stddev, x_mask)