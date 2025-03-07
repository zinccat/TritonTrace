# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_native_group_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_native_group_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (r2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr5 + (r2 + 64*x3), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = -2.0
    tmp10 = tmp8 <= tmp9
    tmp11 = 2.0
    tmp12 = tmp8 >= tmp11
    tmp13 = tmp10 | tmp12
    tmp15 = 0.0
    tmp16 = tl.where(tmp13, tmp15, tmp14)
    tmp17 = tmp16 * tmp0
    tmp18 = tmp17 * tmp5
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tmp16 * tmp5
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp29 = tmp28 * tmp5
    tmp30 = tmp16 * tmp29
    tmp32 = tmp27 * tmp31
    tmp33 = tmp32 - tmp22
    tmp34 = tmp33 * tmp28
    tmp35 = tmp34 * tmp28
    tmp36 = tmp35 * tmp28
    tmp37 = 0.015625
    tmp38 = tmp36 * tmp37
    tmp39 = tmp0 * tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = -tmp38
    tmp42 = tmp41 * tmp31
    tmp43 = tmp27 * tmp28
    tmp44 = tmp43 * tmp37
    tmp45 = tmp42 - tmp44
    tmp46 = tmp40 + tmp45
    tl.store(out_ptr0 + (r2 + 64*x3), tmp8, xmask)
    tl.store(out_ptr3 + (r2 + 64*x3), tmp46, xmask)
