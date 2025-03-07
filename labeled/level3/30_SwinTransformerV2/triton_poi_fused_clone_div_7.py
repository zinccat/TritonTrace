# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_div_7poi_fused_clone_div_7(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    mod_32 = block_index % 32
    div_32_mod_49 = (block_index // 32) % 49
    div_1568_mod_3 = (block_index // 1568) % 3
    div_4704 = block_index // 4704
    div_32 = block_index // 32
    full_index = block_index
    
    input_value0 = tl.load(input_ptr0 + (mod_32 + 32 * div_1568_mod_3 + 288 * div_32_mod_49 + 14112 * div_4704), None)
    input_value1 = tl.load(input_ptr1 + (div_32), None, eviction_policy='evict_last')
    
    epsilon = 1e-12
    max_value = triton_helpers.maximum(input_value1, epsilon)
    
    result = input_value0 / max_value
    tl.store(output_ptr0 + (full_index), result, None)