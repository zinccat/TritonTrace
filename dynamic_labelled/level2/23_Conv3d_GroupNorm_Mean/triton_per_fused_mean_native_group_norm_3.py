# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_native_group_norm_3(in_out_ptr0, in_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 4
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_1 = reduction_index
    input_index_0 = input_index
    loaded_values = tl.load(in_ptr0 + (reduction_index_1 + 4 * input_index_0), input_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(input_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    normalization_factor = (-128) + ((-32) * kernel_size1 * kernel_size1) + 64 * kernel_size0 + 128 * kernel_size1 + ((-64) * kernel_size0 * kernel_size1) + 16 * kernel_size0 * kernel_size1 * kernel_size1
    normalization_factor_float = normalization_factor.to(tl.float32)
    mean_values = summed_values / normalization_factor_float
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (input_index_0), mean_values, input_mask)