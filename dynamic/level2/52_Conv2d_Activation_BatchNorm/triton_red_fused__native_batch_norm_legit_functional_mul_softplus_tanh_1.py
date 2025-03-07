# From: 52_Conv2d_Activation_BatchNorm

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = xindex // 16
    x0 = (xindex % 16)
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))
        tmp1 = 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-2)*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1)))) + 4*x0 + 64*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // ((-2) + ks1)) % ((-2) + ks1))) + x0*ks1*ks1 + ((-64)*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0))) + ((-4)*ks1*x0) + 16*ks1*ks1*((((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) // (4 + ks1*ks1 + ((-4)*ks1))) % ks0)) + (((r2 + x1*(triton_helpers.div_floor_integer(14 + 4*ks0 + ks0*ks1*ks1 + ((-4)*ks0*ks1),  15))) % ((-2) + ks1)))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 20.0
        tmp5 = tmp3 > tmp4
        tmp6 = tl_math.exp(tmp3)
        tmp7 = libdevice.log1p(tmp6)
        tmp8 = tl.where(tmp5, tmp3, tmp7)
        tmp9 = libdevice.tanh(tmp8)
        tmp10 = tmp9 * tmp3
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = 1.0
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp21 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_combine(
            tmp22_mean, tmp22_m2, tmp22_weight,
            tmp19, tmp20, tmp21
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp23, xmask)
    tl.store(out_ptr2 + (x3), tmp24, xmask)
