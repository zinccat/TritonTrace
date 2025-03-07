# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_eq_ge_masked_fill_mul_select_backward_threshold_backward_where_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // ks0)
    x5 = xindex % ks0
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x6 = (xindex // 1568) % ks1
    x7 = (xindex // 32) % ks2
    x2 = (xindex // 1568) % 24
    x3 = (xindex // 37632) % ks3
    x8 = xindex
    tmp3 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1 + (49*x0) + (1568*x6)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x7), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x7), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0 + (32*x2) + (2304*x1) + (112896*x3)), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x5), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x7), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x7), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (x0 + (32*x2) + (2304*x1) + (112896*x3)), xmask, eviction_policy='evict_last')
    tmp0 = x4
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp10 = 1e-12
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tmp8 / tmp11
    tmp13 = tmp9 >= tmp10
    tmp15 = tl.where(tmp13, tmp14, tmp4)
    tmp16 = tmp9 == tmp4
    tmp18 = tmp17 / tmp9
    tmp19 = tl.where(tmp16, tmp4, tmp18)
    tmp20 = tmp15 * tmp19
    tmp21 = tmp12 + tmp20
    tmp22 = tl.where(tmp7, tmp21, tmp4)
    tmp23 = tmp5 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = tmp0 == tmp24
    tmp28 = triton_helpers.maximum(tmp27, tmp10)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp27 >= tmp10
    tmp32 = tl.where(tmp30, tmp31, tmp4)
    tmp33 = tmp27 == tmp4
    tmp35 = tmp34 / tmp27
    tmp36 = tl.where(tmp33, tmp4, tmp35)
    tmp37 = tmp32 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = tl.where(tmp25, tmp38, tmp4)
    tmp40 = tmp23 + tmp39
    tl.store(out_ptr0 + (x8), tmp40, xmask)
