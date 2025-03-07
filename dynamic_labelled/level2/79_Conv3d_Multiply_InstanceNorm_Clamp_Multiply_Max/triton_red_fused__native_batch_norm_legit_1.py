# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_1(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size0, kernel_size1, x_num_elements, r_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    tmp1 = tl.load(in_ptr1 + ((x0 % 16)), x_mask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r1 = r_index
        tmp0 = tl.load(
            in_ptr0 + (r1 + ((-8) * x0) + ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 + kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0)), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, r_offset == 0
        )
        tmp4_mean = tl.where(r_mask & x_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r_mask & x_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r_mask & x_mask, tmp4_weight_next, tmp4_weight)
    
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp4, x_mask)
    
    tmp7 = (
        (tl.full([], 0.0, tl.float64) * (tl.full([], 0.0, tl.float64) >= ((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1))) + 
        ((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)) * 
        (((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)) > (tl.full([], 0.0, tl.float64))))
    )
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp5 / tmp8
    epsilon = 1e-05
    tmp11 = tmp9 + epsilon
    tmp12 = tl.extra.cuda.libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, x_mask)