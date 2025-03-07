# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_0(
    in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 12600
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = x_index % 16
    input_data0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (x3 % 16), None, eviction_policy='evict_last')
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        current_data = tl.load(in_out_ptr0 + (r2 + (12600 * x3)), r_mask, eviction_policy='evict_first', other=0.0)
        updated_data = current_data + input_data0
        multiplied_data = updated_data * input_data1
        broadcasted_data = tl.broadcast_to(multiplied_data, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_data, running_mean, running_m2, running_weight, r_offset == 0
        )
        
        running_mean = tl.where(r_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask, running_weight_next, running_weight)
        
        tl.store(in_out_ptr0 + (r2 + (12600 * x3)), updated_data, r_mask)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(out_ptr0 + (x3), mean, None)
    variance_div = variance / 12600.0
    epsilon = 1e-05
    variance_adjusted = variance_div + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), inv_sqrt_variance, None)