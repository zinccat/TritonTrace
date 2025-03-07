# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_native_group_norm_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_native_group_norm_backward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (r2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr5 + (r2 + 64*x3), xmask, other=0.0)
    tmp12 = tl.load(in_ptr6 + (r2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tl.sigmoid(tmp8)
    tmp11 = tmp8 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp9 * tmp14
    tmp16 = tmp9 * tmp13
    tmp17 = 1.0
    tmp18 = tmp17 - tmp14
    tmp19 = tmp14 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp15 + tmp20
    tmp22 = tmp21 * tmp12
    tmp23 = tmp22 * tmp10
    tmp24 = tmp22 * tmp8
    tmp25 = tmp17 - tmp10
    tmp26 = tmp10 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp23 + tmp27
    tmp29 = tmp28 * tmp0
    tmp30 = tmp29 * tmp5
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tmp28 * tmp5
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp41 = tmp40 * tmp5
    tmp42 = tmp28 * tmp41
    tmp44 = tmp39 * tmp43
    tmp45 = tmp44 - tmp34
    tmp46 = tmp45 * tmp40
    tmp47 = tmp46 * tmp40
    tmp48 = tmp47 * tmp40
    tmp49 = 0.015625
    tmp50 = tmp48 * tmp49
    tmp51 = tmp0 * tmp50
    tmp52 = tmp42 + tmp51
    tmp53 = -tmp50
    tmp54 = tmp53 * tmp43
    tmp55 = tmp39 * tmp40
    tmp56 = tmp55 * tmp49
    tmp57 = tmp54 - tmp56
    tmp58 = tmp52 + tmp57
    tl.store(out_ptr0 + (r2 + 64*x3), tmp8, xmask)
    tl.store(in_out_ptr0 + (r2 + 64*x3), tmp58, xmask)
