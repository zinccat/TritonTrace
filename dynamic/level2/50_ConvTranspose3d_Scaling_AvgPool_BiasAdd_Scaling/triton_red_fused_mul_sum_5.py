# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'ks5': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_5(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1923
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))
        tmp1 = ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-1)*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks3) % ks3))) + ((-1)*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + ((-4)*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4))) + ((-4)*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + 2*ks1*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + 2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks3) % ks3)) + 4*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + 4*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4)) + ((-8)*ks1*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + 8*ks1*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + (((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) % ks3)) + ((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((-1)*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks3) % ks3))) + ((-1)*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + ((-4)*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4))) + ((-4)*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + 2*ks1*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + 2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks3) % ks3)) + 4*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + 4*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4)) + ((-8)*ks1*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0)))) + 8*ks1*ks2*ks2*((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ((-1) + ((-4)*ks2*ks2) + 2*ks1 + 4*ks2 + ((-8)*ks1*ks2) + 8*ks1*ks2*ks2)) % (16*ks0))) + (((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) % ks3)) + ((((r1 + x0*(triton_helpers.div_floor_integer(1922 + ((-16)*ks0) + ((-64)*ks0*ks2*ks2) + 32*ks0*ks1 + 64*ks0*ks2 + ((-128)*ks0*ks1*ks2) + 128*ks0*ks1*ks2*ks2,  1923))) // ks5) % ks4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
