# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_backward_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (4*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x2), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp1
    tmp20 = tmp19 * tmp4
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp8
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp12
    tmp27 = tmp24 + tmp26
    tmp28 = tmp16 - tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp29
    tmp32 = tmp31 * tmp29
    tmp33 = 2.0
    tmp34 = ks0
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 * tmp35
    tmp37 = -1.0
    tmp38 = tmp37 + tmp36
    tmp39 = libdevice.pow(tmp38, tmp33)
    tmp40 = 4.0
    tmp41 = tmp40 * tmp39
    tmp42 = ks1
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp33 * tmp43
    tmp45 = tmp37 + tmp44
    tmp46 = tmp41 * tmp45
    tmp47 = tmp46.to(tl.float64)
    tmp48 = tl.full([1], 1.0, tl.float64)
    tmp49 = tmp48 / tmp47
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp32 * tmp50
    tmp52 = -tmp51
    tmp53 = tmp52 * tmp15
    tmp54 = tmp14 * tmp29
    tmp55 = tmp54 * tmp50
    tmp56 = tmp53 - tmp55
    tl.store(out_ptr1 + (x2), tmp28, xmask)
    tl.store(in_out_ptr0 + (x2), tmp56, xmask)
