# From: 48_Mean_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_1per_fused_mean_1(in_out_ptr0, in_ptr0, kernel_size, total_elements, reduction_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_mod_kernel = x_indices % kernel_size
    x_div_kernel = x_indices // kernel_size
    x_full_indices = x_indices
    loaded_values = tl.load(in_ptr0 + (x_mod_kernel + kernel_size * r2 + 2 * kernel_size * x_div_kernel), x_mask, eviction_policy='evict_last', other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    kernel_size_float = kernel_size.to(tl.float32)
    mean_values = summed_values / kernel_size_float
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_full_indices), mean_values, x_mask)