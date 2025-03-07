# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32', 16: 'i32', 17: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // ks0)
    x3 = xindex % ks0
    x1 = (xindex // ks1) % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (ks0 + x3 + (224*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (ks0 + x3 + (192*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (ks0 + x3 + (160*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (ks0 + x3 + (128*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (ks0 + x3 + (96*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (ks0 + x3 + (64*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (ks0 + x3 + (64*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (ks0 + x3 + (64*ks1*x2)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr10 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = 0.0
    tmp11 = tmp9 <= tmp10
    tmp13 = tl.where(tmp11, tmp10, tmp12)
    tmp16 = tmp14 - tmp15
    tmp18 = 1.00000000000000 / ((64*ks1*ks2) / 64)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tmp13 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp8 + tmp31
    tl.store(out_ptr0 + (x4), tmp32, xmask)
