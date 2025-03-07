# From: 35_GroupNorm_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    block_index = xindex
    group_index = (xindex // 65536)
    channel_index = (xindex // 65536) % 64
    
    # Load input data
    input_data = tl.load(in_ptr0 + (block_index), None)
    mean = tl.load(in_ptr1 + ((group_index // 8)), None, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + ((group_index // 8)), None, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (channel_index), None, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (channel_index), None, eviction_policy='evict_last')
    
    # Normalize
    centered_data = input_data - mean
    inv_std_dev = 1.0 / tl.sqrt(variance / 524288.0 + 1e-05)
    normalized_data = centered_data * inv_std_dev
    
    # Scale and shift
    scaled_data = normalized_data * gamma
    output_data = scaled_data + beta
    
    # Store result
    tl.store(out_ptr0 + (block_index), output_data, None)