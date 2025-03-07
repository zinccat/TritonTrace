# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_hardswish_hardswish_backward_logsumexp_mul_native_group_norm_sub_tanh_tanh_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_hardswish_hardswish_backward_logsumexp_mul_native_group_norm_sub_tanh_tanh_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x5 = xindex // ks0
    x1 = ((xindex // ks1) % 16)
    x4 = (xindex % ks0)
    x7 = xindex // ks2
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 // 2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 // 2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x4 + 4*x7 + x7*ks3*ks3 + ((-4)*ks3*x7)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x4 + 4*x7 + x7*ks3*ks3 + ((-4)*ks3*x7)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = -3.0
    tmp11 = tmp9 < tmp10
    tmp12 = 3.0
    tmp13 = tmp9 <= tmp12
    tmp15 = tmp9 + tmp12
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = 0.16666666666666666
    tmp22 = tmp20 * tmp21
    tmp23 = tmp0 + tmp22
    tmp25 = tmp23 - tmp24
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp14 * tmp26
    tmp28 = 0.3333333333333333
    tmp29 = tmp9 * tmp28
    tmp30 = 0.5
    tmp31 = tmp29 + tmp30
    tmp32 = tmp27 * tmp31
    tmp33 = tl.where(tmp13, tmp32, tmp27)
    tmp34 = tl.where(tmp11, tmp16, tmp33)
    tmp35 = tmp9 * tmp9
    tmp36 = 1.0
    tmp37 = tmp36 - tmp35
    tmp38 = tmp34 * tmp37
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp38, xmask)
