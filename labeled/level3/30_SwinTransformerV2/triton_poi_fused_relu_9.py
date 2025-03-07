# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_9poi_fused_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 86528
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    global_index = index_within_block
    local_index = index_within_block % 512
    output_value = tl.load(in_out_ptr0 + (global_index), valid_indices_mask)
    input_value = tl.load(in_ptr0 + (local_index), valid_indices_mask, eviction_policy='evict_last')
    sum_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, sum_value)
    tl.store(in_out_ptr0 + (global_index), relu_output, valid_indices_mask)