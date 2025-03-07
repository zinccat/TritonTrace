# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_6poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    index_within_block = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_id = (index_within_block // 1024) % 3
    lane_id = index_within_block % 16
    warp_id = (index_within_block // 16) % 64
    wave_id = (index_within_block // 3072) % 8
    grid_id = index_within_block // 24576
    global_index = index_within_block
    
    block_condition = block_id
    tl.full([1], 0, tl.int64)
    condition_true = tl.full([1], 1, tl.int64)
    is_block_zero = block_condition < condition_true
    
    zero_value = 0.0
    zero_tensor = tl.full(zero_value.shape, 0.0, zero_value.dtype)
    result_if_zero = tl.where(is_block_zero, zero_value, zero_tensor)
    
    is_block_nonzero = block_condition >= condition_true
    tl.full([1], 3, tl.int64)
    
    load_index = (warp_id + 64 * lane_id + 1024 * wave_id + 8192 * ((-1) + block_id) + 16384 * grid_id)
    loaded_value = tl.load(in_ptr0 + load_index, is_block_nonzero, eviction_policy='evict_last', other=0.0)
    
    final_result = tl.where(is_block_zero, result_if_zero, loaded_value)
    tl.store(out_ptr0 + global_index, final_result, None)