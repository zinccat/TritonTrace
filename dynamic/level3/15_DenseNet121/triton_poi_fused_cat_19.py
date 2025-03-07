# From: 15_DenseNet121

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0) % 192
    x3 = xindex % ks1
    x4 = (xindex // ks1) % 192
    x5 = (xindex // ks2)
    x6 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (64*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))*x4) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  4))*x4) + (64*x5*((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))) + (128*x5*(triton_helpers.div_floor_integer((-1) + ks3,  4))) + x4), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x3 + (32*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))*((-64) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  4))*((-64) + x4)) + (32*x5*((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))) + (64*x5*(triton_helpers.div_floor_integer((-1) + ks3,  4))) + ((-64) + x4)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x3 + (32*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))*((-96) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  4))*((-96) + x4)) + (32*x5*((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))) + (64*x5*(triton_helpers.div_floor_integer((-1) + ks3,  4))) + ((-96) + x4)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x3 + (32*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))*((-128) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  4))*((-128) + x4)) + (32*x5*((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))) + (64*x5*(triton_helpers.div_floor_integer((-1) + ks3,  4))) + ((-128) + x4)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (x3 + (32*x5) + (((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))*((-160) + x4)) + (2*(triton_helpers.div_floor_integer((-1) + ks3,  4))*((-160) + x4)) + (32*x5*((triton_helpers.div_floor_integer((-1) + ks3,  4))*(triton_helpers.div_floor_integer((-1) + ks3,  4)))) + (64*x5*(triton_helpers.div_floor_integer((-1) + ks3,  4))) + ((-160) + x4)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tl.store(out_ptr0 + (x6), tmp28, xmask)
