# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x1 = xindex // 4
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (((-128)*x1) + ((-8)*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16))) + ((-2)*((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-2) + ks1)) % ((-2) + ks1)))) + 4*((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // (4 + ks1*ks1 + ((-4)*ks1))) % ((-2) + ks0))) + ks1*((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-2) + ks1)) % ((-2) + ks1))) + ks1*ks1*((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // (4 + ks1*ks1 + ((-4)*ks1))) % ((-2) + ks0))) + ((-32)*x1*ks1*ks1) + ((-4)*ks1*((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // (4 + ks1*ks1 + ((-4)*ks1))) % ((-2) + ks0)))) + ((-2)*ks1*ks1*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16))) + 4*ks0*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16)) + 8*ks1*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16)) + 64*ks0*x1 + 128*ks1*x1 + ks0*ks1*ks1*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16)) + ((-64)*ks0*ks1*x1) + ((-4)*ks0*ks1*((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) % 16))) + 16*ks0*x1*ks1*ks1 + ((r2 % ((-2) + ks1)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (8*x1 + (((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) // 2) % 8))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (8*x1 + (((((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)) // 2) % 8))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr3 + ((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr4 + ((((r2 + ((-32)*x0) + ((-8)*x0*ks1*ks1) + 16*ks0*x0 + 32*ks1*x0 + ((-16)*ks0*ks1*x0) + 4*ks0*x0*ks1*ks1) // ((-8) + ((-2)*ks1*ks1) + 4*ks0 + 8*ks1 + ks0*ks1*ks1 + ((-4)*ks0*ks1))) % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = ((tl.full([], 0.0, tl.float64)) * ((tl.full([], 0.0, tl.float64)) >= ((-16) + ((-4)*ks1*ks1) + 8*ks0 + 16*ks1 + ((-8)*ks0*ks1) + 2*ks0*ks1*ks1)) + ((-16) + ((-4)*ks1*ks1) + 8*ks0 + 16*ks1 + ((-8)*ks0*ks1) + 2*ks0*ks1*ks1) * (((-16) + ((-4)*ks1*ks1) + 8*ks0 + 16*ks1 + ((-8)*ks0*ks1) + 2*ks0*ks1*ks1) > (tl.full([], 0.0, tl.float64))))
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 / tmp5
        tmp7 = 1e-05
        tmp8 = tmp6 + tmp7
        tmp9 = libdevice.rsqrt(tmp8)
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
