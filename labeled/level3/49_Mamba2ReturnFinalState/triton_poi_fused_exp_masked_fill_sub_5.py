# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_exp_masked_fill_sub_5poi_fused_exp_masked_fill_sub_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    remainder_mod_9 = index_within_block % 9
    quotient_div_3 = index_within_block // 3
    remainder_mod_3 = index_within_block % 3
    quotient_div_9 = index_within_block // 9
    absolute_index = index_within_block

    input_mask = tl.load(in_ptr0 + (remainder_mod_9), valid_mask, eviction_policy='evict_last').to(tl.int1)
    input_value1 = tl.load(in_ptr1 + (quotient_div_3), valid_mask, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr1 + (remainder_mod_3 + 3 * quotient_div_9), valid_mask, eviction_policy='evict_last')

    difference = input_value1 - input_value2
    negative_infinity = float("-inf")
    masked_difference = tl.where(input_mask, negative_infinity, difference)

    exp_result = tl.math.exp(masked_difference)
    tl.store(out_ptr0 + (absolute_index), exp_result, valid_mask)