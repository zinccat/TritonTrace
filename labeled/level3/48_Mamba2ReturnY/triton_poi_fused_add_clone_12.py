# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clone_12poi_fused_add_clone_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_index = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index = block_index
    block_64_index = (index // 64) % 64
    block_4096_index = (index // 4096) % 8
    block_32768_index = (index // 32768) % 2
    block_65536_index = index // 65536
    index_mod_64 = index % 64
    index_div_32768 = index // 32768
    
    input0_value = tl.load(in_ptr0 + (index), None)
    input1_value = tl.load(in_ptr1 + (index), None)
    input2_value = tl.load(in_ptr2 + (block_64_index + 64*block_32768_index + 128*block_4096_index + 1024*block_65536_index), None, eviction_policy='evict_last')
    
    exp_value = tl.math.exp(input2_value)
    scaled_input1 = input1_value * exp_value
    result_value = input0_value + scaled_input1
    
    store_index = (index_mod_64 + 64*block_4096_index + 512*block_64_index + 32768*index_div_32768)
    tl.store(out_ptr0 + (store_index), result_value, None)