# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_exp_masked_fill_sub_9poi_fused_exp_masked_fill_sub_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    mask = index_within_block < xnumel
    remainder_by_9 = index_within_block % 9
    quotient_by_3 = index_within_block // 3
    remainder_by_3 = index_within_block % 3
    quotient_by_9 = index_within_block // 9
    global_index = index_within_block

    tmp0 = tl.load(in_ptr0 + (remainder_by_9), mask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (quotient_by_3), mask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (remainder_by_3 + 3 * quotient_by_9), mask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    negative_infinity = float("-inf")
    tmp5 = tl.where(tmp0, negative_infinity, tmp3)
    tmp6 = tl.math.exp(tmp5)
    tl.store(out_ptr0 + (global_index), tmp6, mask)