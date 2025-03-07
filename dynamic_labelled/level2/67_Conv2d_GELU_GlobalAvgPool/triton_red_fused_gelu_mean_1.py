# From: 67_Conv2d_GELU_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_mean_1red_fused_gelu_mean_1(in_out_ptr0, in_ptr0, kernel_size, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        temp_load = tl.load(in_ptr0 + (r1 + 4*x0 + x0*kernel_size*kernel_size + ((-4)*kernel_size*x0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        half = 0.5
        temp_half = temp_load * half
        sqrt_half = 0.7071067811865476
        temp_sqrt_half = temp_load * sqrt_half
        erf_result = tl.extra.cuda.libdevice.erf(temp_sqrt_half)
        one = 1.0
        temp_erf_sum = erf_result + one
        temp_gelu = temp_half * temp_erf_sum
        temp_broadcast = tl.broadcast_to(temp_gelu, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulate, temp_sum)
    
    temp_reduce = tl.sum(temp_sum, 1)[:, None]
    temp_factor = 4 + kernel_size*kernel_size + ((-4)*kernel_size)
    temp_factor_float = temp_factor.to(tl.float32)
    temp_mean = temp_reduce / temp_factor_float
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), temp_mean, x_mask)