# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (((-8)*x0) + ((-2)*(((r1 // ((-2) + ks1)) % ((-2) + ks1)))) + 4*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ks1*(((r1 // ((-2) + ks1)) % ((-2) + ks1))) + ks1*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ((-4)*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1)))) + ((-2)*x0*ks1*ks1) + 4*ks0*x0 + 8*ks1*x0 + ks0*x0*ks1*ks1 + ((-4)*ks0*ks1*x0) + ((r1 % ((-2) + ks1)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + (((-8)*x0) + ((-2)*(((r1 // ((-2) + ks1)) % ((-2) + ks1)))) + 4*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ks1*(((r1 // ((-2) + ks1)) % ((-2) + ks1))) + ks1*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ((-4)*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1)))) + ((-2)*x0*ks1*ks1) + 4*ks0*x0 + 8*ks1*x0 + ks0*x0*ks1*ks1 + ((-4)*ks0*ks1*x0) + ((r1 % ((-2) + ks1)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (((-8)*x0) + ((-2)*(((r1 // ((-2) + ks1)) % ((-2) + ks1)))) + 4*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ks1*(((r1 // ((-2) + ks1)) % ((-2) + ks1))) + ks1*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ((-4)*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1)))) + ((-2)*x0*ks1*ks1) + 4*ks0*x0 + 8*ks1*x0 + ks0*x0*ks1*ks1 + ((-4)*ks0*ks1*x0) + ((r1 % ((-2) + ks1)))), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp20 = tl.load(in_ptr3 + (((-8)*x0) + ((-2)*(((r1 // ((-2) + ks1)) % ((-2) + ks1)))) + 4*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ks1*(((r1 // ((-2) + ks1)) % ((-2) + ks1))) + ks1*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1))) + ((-4)*ks1*(triton_helpers.div_floor_integer(r1,  4 + ks1*ks1 + ((-4)*ks1)))) + ((-2)*x0*ks1*ks1) + 4*ks0*x0 + 8*ks1*x0 + ks0*x0*ks1*ks1 + ((-4)*ks0*ks1*x0) + ((r1 % ((-2) + ks1)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp3 = tmp0 == tmp1
        tmp4 = triton_helpers.minimum(tmp0, tmp1)
        tmp5 = tmp4 >= tmp1
        tmp6 = 1.0
        tmp7 = tmp4 <= tmp6
        tmp8 = tmp5 & tmp7
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 1.25
        tmp13 = tmp11 * tmp12
        tmp14 = tmp9 * tmp13
        tmp15 = tl.where(tmp8, tmp14, tmp1)
        tmp16 = 0.5
        tmp17 = tmp15 * tmp16
        tmp18 = tl.where(tmp3, tmp17, tmp15)
        tmp19 = tl.where(tmp2, tmp1, tmp18)
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr1 + (x0), tmp26, xmask)
