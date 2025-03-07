# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // ks0) % 320
    x0 = xindex % ks1
    x1 = (xindex // ks1) % ks1
    x3 = (xindex // ks2)
    x4 = xindex % ks0
    x5 = (xindex // ks0)
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + x1 + (128*x3) + (x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))) + (((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2)))*x2) + (2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))*x2) + (128*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2)))) + (256*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))) + x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + (ks0*((-128) + x2)) + (32*ks0*x3)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + (ks0*((-160) + x2)) + (32*ks0*x3)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 224, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x4 + (ks0*((-192) + x2)) + (32*ks0*x3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 256, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (x4 + (ks0*((-224) + x2)) + (32*ks0*x3)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 288, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr5 + (x4 + (ks0*((-256) + x2)) + (32*ks0*x3)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 320, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr6 + (x4 + (ks0*((-288) + x2)) + (32*ks0*x3)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp14, tmp15, tmp37)
    tmp39 = tl.where(tmp9, tmp10, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x0 + x1 + x5 + (x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))) + (x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2)))) + (2*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks3,  4)),  2)))), tmp40, xmask)
