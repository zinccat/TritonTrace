# From: 100_HingeLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_mean_mul_rsub_0(in_out_ptr0, in_ptr0, in_ptr1, ks0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_mask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < rnumel
        r_indices_clamped = r_indices
        input0_values = tl.load(in_ptr0 + (r_indices_clamped), r_mask, eviction_policy='evict_first', other=0.0)
        input1_values = tl.load(in_ptr1 + (r_indices_clamped), r_mask, eviction_policy='evict_first', other=0.0)
        elementwise_product = input0_values * input1_values
        one_minus_product = 1.0 - elementwise_product
        zero = 0.0
        clamped_values = triton_helpers.maximum(one_minus_product, zero)
        broadcasted_clamped = tl.broadcast_to(clamped_values, [XBLOCK, RBLOCK])
        temp_sum += tl.where(r_mask, broadcasted_clamped, temp_sum)
    
    sum_over_r = tl.sum(temp_sum, 1)[:, None]
    ks0_float = ks0.to(tl.float32)
    mean_values = sum_over_r / ks0_float
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), mean_values, None)