# From: 5_AlexNet

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i8', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks0
    x2 = (xindex // ks1)
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (ks2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + ks2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (2 + ks2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((2*ks2) + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + (2*ks2) + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + (2*ks2) + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = tl.full([1], 1, tl.int8)
    tmp19 = tl.full([1], 0, tl.int8)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = tl.full([1], 2, tl.int8)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = tl.full([1], 3, tl.int8)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = tl.full([1], 4, tl.int8)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = tl.full([1], 5, tl.int8)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = tl.full([1], 6, tl.int8)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = tl.full([1], 7, tl.int8)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = tl.full([1], 8, tl.int8)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x0 + x1 + x2 + (x1*(triton_helpers.div_floor_integer((-3) + ks2,  2))) + (x2*((triton_helpers.div_floor_integer((-3) + ks2,  2))*(triton_helpers.div_floor_integer((-3) + ks2,  2)))) + (2*x2*(triton_helpers.div_floor_integer((-3) + ks2,  2)))), tmp16, xmask)
    tl.store(out_ptr1 + (x0 + x1 + x2 + (x1*(triton_helpers.div_floor_integer((-3) + ks2,  2))) + (x2*((triton_helpers.div_floor_integer((-3) + ks2,  2))*(triton_helpers.div_floor_integer((-3) + ks2,  2)))) + (2*x2*(triton_helpers.div_floor_integer((-3) + ks2,  2)))), tmp41, xmask)
