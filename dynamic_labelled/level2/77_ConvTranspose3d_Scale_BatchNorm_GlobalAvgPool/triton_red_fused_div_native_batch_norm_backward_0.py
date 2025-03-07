# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_div_native_batch_norm_backward_0(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 32
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_4 = reduction_index // kernel_size
        temp_load = tl.load(in_ptr0 + (input_0 + 32 * reduction_4), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp_kernel_size = kernel_size
        temp_kernel_size_float = temp_kernel_size.to(tl.float32)
        temp_div = temp_load / temp_kernel_size_float
        temp_broadcast = tl.broadcast_to(temp_div, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_accumulate, temp_sum)
    
    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_0), temp_result, input_mask)