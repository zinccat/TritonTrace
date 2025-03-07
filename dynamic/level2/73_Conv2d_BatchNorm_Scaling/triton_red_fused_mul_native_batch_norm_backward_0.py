# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_batch_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 16
    x0 = (xindex % 16)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))
        tmp1 = 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-2)*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1)))) + 4*x0 + 64*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1))) + x0*ks1*ks1 + ((-64)*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0))) + ((-4)*ks1*x0) + 16*ks1*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) % ((-2) + ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 2.0
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr1 + (((-2)*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1)))) + 4*x0 + 64*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1))) + x0*ks1*ks1 + ((-64)*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0))) + ((-4)*ks1*x0) + 16*ks1*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) % ((-2) + ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp5 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
