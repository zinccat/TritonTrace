# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_native_batch_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 32
    x0 = (xindex % 32)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))
        tmp1 = 496 + ((-1984)*ks0) + 1984*ks0*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-1)*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // ((-1) + 2*ks0)) % ((-1) + 2*ks0)))) + 31*x0 + 992*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16)) + ((-3968)*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16))) + ((-124)*ks0*x0) + ((-4)*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31))) + 2*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // ((-1) + 2*ks0)) % ((-1) + 2*ks0))) + 4*ks0*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31)) + 124*x0*ks0*ks0 + 3968*ks0*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16)) + (((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) % ((-1) + 2*ks0))) + ((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 32*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.broadcast_to(31 + ((-124)*ks0) + 124*ks0*ks0, [XBLOCK, RBLOCK])
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 / tmp6
        tmp8 = tmp3 + tmp7
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (((-1)*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // ((-1) + 2*ks0)) % ((-1) + 2*ks0)))) + 31*x0 + 992*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16)) + ((-3968)*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16))) + ((-124)*ks0*x0) + ((-4)*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31))) + 2*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // ((-1) + 2*ks0)) % ((-1) + 2*ks0))) + 4*ks0*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31)) + 124*x0*ks0*ks0 + 3968*ks0*ks0*((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (31 + ((-124)*ks0) + 124*ks0*ks0)) % 16)) + (((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) % ((-1) + 2*ks0))) + ((((r2 + 46*x1 + x1*(triton_helpers.div_floor_integer(((-1984)*ks0) + 1984*ks0*ks0,  11))) // (1 + ((-4)*ks0) + 4*ks0*ks0)) % 31))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp8 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp21, xmask)
