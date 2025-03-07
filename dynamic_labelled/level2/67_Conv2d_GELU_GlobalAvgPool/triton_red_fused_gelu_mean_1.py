# From: 67_Conv2d_GELU_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_mean_1(in_out_ptr0, in_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices
        input_data = tl.load(
            in_ptr0 + (r_indices_flat + 4 * x_indices_flat + x_indices_flat * kernel_size * kernel_size + ((-4) * kernel_size * x_indices_flat)),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        
        half = 0.5
        scaled_input = input_data * half
        sqrt_half = 0.7071067811865476
        scaled_sqrt_input = input_data * sqrt_half
        erf_result = tl.extra.cuda.libdevice.erf(scaled_sqrt_input)
        one = 1.0
        erf_plus_one = erf_result + one
        gelu_result = scaled_input * erf_plus_one
        broadcast_gelu = tl.broadcast_to(gelu_result, [XBLOCK, RBLOCK])
        temp_sum += tl.where(r_mask & x_mask, broadcast_gelu, temp_sum)
    
    sum_over_reduction = tl.sum(temp_sum, 1)[:, None]
    kernel_size_adjustment = 4 + kernel_size * kernel_size + ((-4) * kernel_size)
    kernel_size_adjustment_float = kernel_size_adjustment.to(tl.float32)
    mean_result = sum_over_reduction / kernel_size_adjustment_float
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_indices_flat), mean_result, x_mask)