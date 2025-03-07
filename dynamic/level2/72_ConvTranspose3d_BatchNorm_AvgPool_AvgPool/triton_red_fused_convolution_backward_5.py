# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_backward_5(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 249)
    x1 = xindex // 249
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))
        tmp1 = ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((-1)*x1) + ((-1)*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ks2) % ks2))) + ((-16)*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ((-1) + ((-12)*ks1*ks1) + 6*ks1 + 8*ks1*ks1*ks1)) % ks0))) + ((-192)*ks1*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ((-1) + ((-12)*ks1*ks1) + 6*ks1 + 8*ks1*ks1*ks1)) % ks0))) + ((-12)*x1*ks1*ks1) + ((-4)*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ks3) % ks2))) + 2*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ks2) % ks2)) + 4*ks1*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ks3) % ks2)) + 6*ks1*x1 + 8*x1*ks1*ks1*ks1 + 96*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ((-1) + ((-12)*ks1*ks1) + 6*ks1 + 8*ks1*ks1*ks1)) % ks0)) + 128*ks1*ks1*ks1*((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ((-1) + ((-12)*ks1*ks1) + 6*ks1 + 8*ks1*ks1*ks1)) % ks0)) + (((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) % ks2)) + ((((r2 + x0*(triton_helpers.div_floor_integer(248 + ((-1)*ks0) + ((-12)*ks0*ks1*ks1) + 6*ks0*ks1 + 8*ks0*ks1*ks1*ks1,  249))) // ks3) % ks2))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
