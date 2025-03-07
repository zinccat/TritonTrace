# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (2*(((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) // (2 + ks0)) % (2 + ks0))) + 4*(((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) // (4 + ks0*ks0 + 4*ks0)) % (2 + ks1))) + 8*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 128*x1 + ks0*(((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) // (2 + ks0)) % (2 + ks0))) + ks0*ks0*(((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) // (4 + ks0*ks0 + 4*ks0)) % (2 + ks1))) + 2*ks0*ks0*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 4*ks0*(((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) // (4 + ks0*ks0 + 4*ks0)) % (2 + ks1))) + 4*ks1*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 8*ks0*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 32*x1*ks0*ks0 + 64*ks1*x1 + 128*ks0*x1 + ks1*ks0*ks0*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 4*ks0*ks1*((((r2 + 32*x0 + 8*x0*ks0*ks0 + 16*ks1*x0 + 32*ks0*x0 + 4*ks1*x0*ks0*ks0 + 16*ks0*ks1*x0) // (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1)) % 16)) + 16*ks1*x1*ks0*ks0 + 64*ks0*ks1*x1 + ((((r2 % (8 + 2*ks0*ks0 + 4*ks1 + 8*ks0 + ks1*ks0*ks0 + 4*ks0*ks1))) % (2 + ks0)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
    tl.store(out_ptr2 + (x3), tmp6, xmask)
