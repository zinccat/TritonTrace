# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1(in_out_ptr0, in_ptr0, out_ptr0, kernel_size_z, kernel_size_y, kernel_size_x, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
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
        r_z = (r_index % kernel_size_z)
        r_y = r_index // kernel_size_z
        
        input_index = (
            (-1) * r_y + 
            (-1) * ((r_z // ((-1) + 2 * kernel_size_x)) % ((-1) + 2 * kernel_size_x)) + 
            (-4) * x0 + 
            (-16) * x0 * kernel_size_x * kernel_size_x + 
            (-4) * kernel_size_x * (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            (-4) * r_y * kernel_size_x * kernel_size_x + 
            2 * kernel_size_y * r_y + 
            2 * kernel_size_x * ((r_z // ((-1) + 2 * kernel_size_x)) % ((-1) + 2 * kernel_size_x)) + 
            4 * kernel_size_x * r_y + 
            4 * kernel_size_x * kernel_size_x * (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            8 * kernel_size_y * x0 + 
            16 * kernel_size_x * x0 + 
            (-32) * kernel_size_y * kernel_size_x * x0 + 
            (-8) * kernel_size_y * kernel_size_x * r_y + 
            8 * kernel_size_y * r_y * kernel_size_x * kernel_size_x + 
            32 * kernel_size_y * x0 * kernel_size_x * kernel_size_x + 
            (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            (r_z % ((-1) + 2 * kernel_size_x))
        )
        
        input_data = tl.load(in_ptr0 + input_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        sigmoid_data = tl.sigmoid(input_data)
        weighted_input = sigmoid_data * input_data
        broadcasted_weighted_input = tl.broadcast_to(weighted_input, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_weighted_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight_sum = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(out_ptr0 + (x0), mean, x_mask)
    
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std, x_mask)