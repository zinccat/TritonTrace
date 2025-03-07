# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_backward_110', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 96) % 3136
    x2 = (xindex // 301056)
    x3 = xindex
    x0 = xindex % 96
    x4 = (xindex // 96)
    x5 = xindex % 301056
    tmp0 = tl.load(in_ptr0 + (x1 + (392*x2*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp15 = tl.load(in_ptr3 + (x3), None)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x5 + (37632*x2*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))), None)
    tmp24 = tl.load(in_ptr7 + (x4), None, eviction_policy='evict_last')
    tmp2 = (((x3 // 96) % 3136) // 56) % 2
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 == tmp3
    tmp5 = (x1 % 56) % 2
    tmp6 = tmp5 == tmp3
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr2 + (x0 + (384*((x1 % 56) // 2)) + (10752*(x1 // 112)) + (301056*x2)), tmp7, other=0.0)
    tmp9 = 0.0
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tl.where(tmp4, tmp12, tmp9)
    tmp14 = tmp1 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = 96.0
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 - tmp21
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tmp27 = tmp0 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
