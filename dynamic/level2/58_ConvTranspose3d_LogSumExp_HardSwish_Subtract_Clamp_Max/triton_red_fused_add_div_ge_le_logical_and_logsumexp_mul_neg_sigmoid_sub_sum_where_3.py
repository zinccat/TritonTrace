# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 16
    x0 = (xindex % 16)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))
        tmp1 = ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-1)*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + ((-1)*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ((-1) + 2*ks2)) % ((-1) + 2*ks2)))) + ((-4)*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1)))) + ((-4)*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + 2*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + 2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ((-1) + 2*ks2)) % ((-1) + 2*ks2))) + 4*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + 4*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1))) + ((-8)*ks1*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + 8*ks1*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) % ((-1) + 2*ks2))) + ((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tmp3 * tmp6
        tmp8 = 0.16666666666666666
        tmp9 = tmp7 * tmp8
        tmp10 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = -1.0
        tmp13 = tmp11 >= tmp12
        tmp14 = 1.0
        tmp15 = tmp11 <= tmp14
        tmp16 = tmp13 & tmp15
        tmp17 = tl.load(in_ptr2 + (((-1)*x0) + ((-1)*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ((-1) + 2*ks2)) % ((-1) + 2*ks2)))) + ((-16)*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + ((-64)*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + ((-4)*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1)))) + ((-4)*x0*ks2*ks2) + 2*ks1*x0 + 2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ((-1) + 2*ks2)) % ((-1) + 2*ks2))) + 4*ks2*x0 + 4*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1))) + 32*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + 64*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + ((-128)*ks1*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0))) + ((-8)*ks1*ks2*x0) + 8*ks1*x0*ks2*ks2 + 128*ks1*ks2*ks2*((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // ks3) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) % ((-1) + 2*ks2))) + ((((r2 + x1*(triton_helpers.div_floor_integer(122 + ((-1)*ks0) + ((-4)*ks0*ks2*ks2) + 2*ks0*ks1 + 4*ks0*ks2 + ((-8)*ks0*ks1*ks2) + 8*ks0*ks1*ks2*ks2,  123))) // (1 + ((-4)*ks2) + 4*ks2*ks2)) % ((-1) + 2*ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = 0.0
        tmp19 = tl.where(tmp16, tmp17, tmp18)
        tmp20 = -tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
