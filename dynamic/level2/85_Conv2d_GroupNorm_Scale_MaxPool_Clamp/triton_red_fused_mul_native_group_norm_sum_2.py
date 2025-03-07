# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_group_norm_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_group_norm_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 16
    x0 = (xindex % 16)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))
        tmp1 = 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-2)*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ks2) % ks2))) + 4*x0 + 64*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ks2) % ks2)) + x0*ks1*ks1 + ((-64)*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0))) + ((-4)*ks1*x0) + 16*ks1*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) % ks2))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (((-2)*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ks2) % ks2))) + 4*x0 + 64*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ks2) % ks2)) + x0*ks1*ks1 + ((-64)*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0))) + ((-4)*ks1*x0) + 16*ks1*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) % ks2))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
