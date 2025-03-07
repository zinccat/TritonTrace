# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 524288, 'r': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*(((r2 + ks0*ks1*x1) % (8192*ks0*ks1)))), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + 64*(((r2 + ks0*ks1*x1) % (8192*ks0*ks1)))), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr2 + (x0 + 64*(((r2 + ks0*ks1*x1) % (8192*ks0*ks1)))), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (((r2 + ks0*ks1*x1) % (8192*ks0*ks1))), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr4 + (((r2 + ks0*ks1*x1) % (8192*ks0*ks1))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 * tmp1
        tmp4 = 0.7071067811865476
        tmp5 = tmp3 * tmp4
        tmp6 = libdevice.erf(tmp5)
        tmp7 = tmp6 + tmp1
        tmp8 = 0.5
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp3
        tmp11 = -0.5
        tmp12 = tmp10 * tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = 0.3989422804014327
        tmp15 = tmp13 * tmp14
        tmp16 = tmp3 * tmp15
        tmp17 = tmp9 + tmp16
        tmp18 = tmp2 * tmp17
        tmp21 = tmp19 - tmp20
        tmp23 = tmp21 * tmp22
        tmp24 = tmp18 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp28 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, None)
    tl.store(out_ptr1 + (x3), tmp29, None)
