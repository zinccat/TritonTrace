# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x6 = xindex // 34848
    x7 = ((xindex // 4356) % 64)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x7), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4), xmask)
    tmp14 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = 0.7071067811865476
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.erf(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp7 * tmp12
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 * tmp1
    tmp20 = tmp19 * tmp1
    tmp21 = tmp20 * tmp1
    tmp22 = 2.869605142332415e-05
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 * tmp23
    tmp25 = tmp4 + tmp24
    tmp26 = -tmp23
    tmp27 = tmp26 * tmp15
    tmp28 = tmp14 * tmp1
    tmp29 = tmp28 * tmp22
    tmp30 = tmp27 - tmp29
    tmp31 = tmp25 + tmp30
    tmp32 = tmp12 * tmp6
    tmp33 = tmp5 * tmp5
    tmp34 = -0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tl_math.exp(tmp35)
    tmp37 = 0.3989422804014327
    tmp38 = tmp36 * tmp37
    tmp39 = tmp5 * tmp38
    tmp40 = tmp32 + tmp39
    tmp41 = tmp31 * tmp40
    tl.store(in_out_ptr0 + (x4), tmp41, xmask)
