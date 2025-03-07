# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 1024
    x4 = xindex // 128
    x3 = (xindex % 1024)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x5), xmask)
    tmp19 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (x5 // 128), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x5 // 128), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x5 // 128), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (x5 // 128), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = 0.0009765625
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 * tmp8
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = 0.7071067811865476
    tmp14 = tmp10 * tmp13
    tmp15 = libdevice.erf(tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp12 * tmp17
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 - tmp22
    tmp24 = tmp23 * tmp6
    tmp25 = tmp24 * tmp6
    tmp26 = tmp25 * tmp6
    tmp27 = 0.0078125
    tmp28 = tmp26 * tmp27
    tmp29 = tmp18 * tmp28
    tmp30 = tmp9 + tmp29
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 - tmp34
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37 * tmp36
    tmp39 = tmp38 * tmp36
    tmp40 = tmp39 * tmp27
    tmp41 = -tmp40
    tmp42 = tmp41 * tmp32
    tmp43 = tmp31 * tmp36
    tmp44 = tmp43 * tmp27
    tmp45 = tmp42 - tmp44
    tmp46 = tmp30 + tmp45
    tmp47 = tmp17 * tmp11
    tmp48 = tmp10 * tmp10
    tmp49 = -0.5
    tmp50 = tmp48 * tmp49
    tmp51 = tl_math.exp(tmp50)
    tmp52 = 0.3989422804014327
    tmp53 = tmp51 * tmp52
    tmp54 = tmp10 * tmp53
    tmp55 = tmp47 + tmp54
    tmp56 = tmp46 * tmp55
    tl.store(in_out_ptr0 + (x5), tmp56, xmask)
