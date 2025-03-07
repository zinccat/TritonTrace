# From: 9_ResNet18

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32', 20: 'i32', 21: 'i32', 22: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x5 = (xindex // ks0)
    x1 = (xindex // ks1) % 512
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x5), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr13 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr14 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr15 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = ks0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tmp8 = tl.where(tmp3, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 1.00000000000000 / (((512*ks2) + (512*ks2*((triton_helpers.div_floor_integer((-1) + ks3,  32))*(triton_helpers.div_floor_integer((-1) + ks3,  32)))) + (1024*ks2*(triton_helpers.div_floor_integer((-1) + ks3,  32)))) / 512)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp14 * tmp21
    tmp23 = tmp11 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 * tmp17
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp11 - tmp35
    tmp38 = tmp37 * tmp17
    tmp39 = tmp36 - tmp38
    tmp41 = tmp19 * tmp40
    tmp42 = tmp26 * tmp41
    tmp44 = tmp32 * tmp43
    tmp45 = tmp39 * tmp44
    tl.store(in_out_ptr0 + (x3), tmp42, xmask)
    tl.store(in_out_ptr1 + (x3), tmp45, xmask)
