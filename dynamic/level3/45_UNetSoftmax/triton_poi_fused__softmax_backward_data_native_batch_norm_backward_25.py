# From: 45_UNetSoftmax

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_native_batch_norm_backward_25', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x5 = (xindex // ks0)
    x2 = (xindex // ks1) % 512
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x5), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = libdevice.fma(tmp1, tmp2, tmp4)
    tmp8 = tmp6 - tmp7
    tmp10 = 1.00000000000000 / ((16384*(ks2 // 16)*(ks3 // 16)) / 512)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x4), tmp23, None)
