# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex // 64
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 1024*r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr3 + (x0 + 1024*r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr4 + (x3 + 16*r1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr5 + (x3 + 16*r1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tl.sigmoid(tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = tmp0 * tmp5
        tmp9 = 1.0
        tmp10 = tmp9 - tmp6
        tmp11 = tmp6 * tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tmp7 + tmp12
        tmp14 = tmp13 * tmp3
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp18 = tmp13 * tmp4
        tmp19 = tmp18 * tmp2
        tmp20 = tmp18 * tmp1
        tmp21 = tmp9 - tmp2
        tmp22 = tmp2 * tmp21
        tmp23 = tmp20 * tmp22
        tmp24 = tmp19 + tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp24 * tmp27
        tmp29 = tmp26 - tmp28
        tmp31 = tmp29 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
        tmp35 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask & xmask, tmp37, _tmp36)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp33, xmask)
    tl.store(out_ptr2 + (x0), tmp36, xmask)
