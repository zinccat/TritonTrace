# From: 97_CosineSimilarityLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_rsub_1red_fused_mean_rsub_1(in_out_ptr0, in_ptr0, ks0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_base = tl.arange(0, RBLOCK)[None, :]
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_values = r_index
        loaded_values = tl.load(in_ptr0 + (r_values), r_mask, eviction_policy='evict_first', other=0.0)
        subtracted_values = 1.0 - loaded_values
        broadcasted_values = tl.broadcast_to(subtracted_values, [XBLOCK, RBLOCK])
        temp_sum = tl.where(r_mask, temp_sum + broadcasted_values, temp_sum)

    summed_values = tl.sum(temp_sum, 1)[:, None]
    kernel_size = ks0.to(tl.float32)
    mean_values = summed_values / kernel_size
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_values, None)