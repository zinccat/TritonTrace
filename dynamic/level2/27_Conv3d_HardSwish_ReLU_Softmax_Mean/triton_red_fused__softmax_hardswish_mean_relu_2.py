# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_hardswish_mean_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_hardswish_mean_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 16
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + ((-8)*x3) + ((-2)*x3*ks1*ks1) + 4*ks0*x3 + 8*ks1*x3 + ks0*x3*ks1*ks1 + ((-4)*ks0*ks1*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r2 + ((-8)*x1) + ((-2)*x1*ks1*ks1) + 4*ks0*x1 + 8*ks1*x1 + ks0*x1*ks1*ks1 + ((-4)*ks0*ks1*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2 + ((-8)*x1) + ((-2)*x1*ks1*ks1) + 4*ks0*x1 + 8*ks1*x1 + ks0*x1*ks1*ks1 + ((-4)*ks0*ks1*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 3.0
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = 6.0
        tmp6 = triton_helpers.minimum(tmp4, tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = 0.16666666666666666
        tmp9 = tmp7 * tmp8
        tmp10 = tl.full([1, 1], 0, tl.int32)
        tmp11 = triton_helpers.maximum(tmp10, tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tl_math.exp(tmp13)
        tmp16 = tmp14 / tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = ks2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp18 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)
