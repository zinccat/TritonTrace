# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0) % 512
    x3 = xindex % ks1
    x4 = (xindex // ks1) % 512
    x5 = (xindex // ks2)
    x6 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (192*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))*x4) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  16))*x4) + (192*x5*((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))) + (384*x5*(triton_helpers.div_floor_integer((-1) + ks3,  16))) + x4), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x4), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 400, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x3 + (208*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))*((-192) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  16))*((-192) + x4)) + (208*x5*((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))) + (416*x5*(triton_helpers.div_floor_integer((-1) + ks3,  16))) + ((-192) + x4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-192) + x4), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 448, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (x3 + (48*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))*((-400) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  16))*((-400) + x4)) + (48*x5*((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))) + (96*x5*(triton_helpers.div_floor_integer((-1) + ks3,  16))) + ((-400) + x4)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr5 + ((-400) + x4), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 512, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr6 + (x3 + (64*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))*((-448) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  16))*((-448) + x4)) + (64*x5*((triton_helpers.div_floor_integer((-1) + ks3,  16))*(triton_helpers.div_floor_integer((-1) + ks3,  16)))) + (128*x5*(triton_helpers.div_floor_integer((-1) + ks3,  16))) + ((-448) + x4)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-448) + x4), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp28, tmp33, tmp34)
    tmp36 = tl.where(tmp22, tmp27, tmp35)
    tmp37 = tl.where(tmp13, tmp18, tmp36)
    tmp38 = tl.where(tmp4, tmp9, tmp37)
    tl.store(out_ptr0 + (x6), tmp38, xmask)
