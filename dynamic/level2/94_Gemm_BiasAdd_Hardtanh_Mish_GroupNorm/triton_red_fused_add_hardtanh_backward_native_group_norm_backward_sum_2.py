# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardtanh_backward_native_group_norm_backward_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_hardtanh_backward_native_group_norm_backward_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp2 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    x1 = xindex // 32
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x3 + 1024*r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x3 + 1024*r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr3 + (x1 + 32*r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x1 + 32*r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr5 + (x3 + 1024*r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = -1.0
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = 1.0
        tmp7 = triton_helpers.minimum(tmp5, tmp6)
        tmp8 = 20.0
        tmp9 = tmp7 > tmp8
        tmp10 = tl_math.exp(tmp7)
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = tl.where(tmp9, tmp7, tmp11)
        tmp13 = libdevice.tanh(tmp12)
        tmp14 = tmp7 * tmp13
        tmp15 = tmp0 * tmp14
        tmp17 = tmp0 * tmp16
        tmp18 = tmp15 - tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tmp3 <= tmp4
        tmp25 = tmp3 >= tmp6
        tmp26 = tmp24 | tmp25
        tmp28 = 0.0
        tmp29 = tl.where(tmp26, tmp28, tmp27)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tmp33 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
    tl.store(out_ptr2 + (x3), tmp34, xmask)
