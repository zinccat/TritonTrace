# From: 29_SwinMLP

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_78', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*((x0 % 56) % 7)) + (224*((x0 // 56) % 7)) + (1568*(r2 // 32)) + (4704*((x0 % 56) // 7)) + (37632*(x0 // 392)) + (301056*x1) + (r2 % 32)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (3136*r2) + (301056*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp32 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_out_ptr0 + (r2 + (96*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr0 + ((32*((x0 % 56) % 7)) + (224*((x0 // 56) % 7)) + (1568*(r2 // 32)) + (4704*((x0 % 56) // 7)) + (37632*(x0 // 392)) + (301056*x1) + (r2 % 32)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr2 + (x0 + (3136*r2) + (301056*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr5 + (x0 + (3136*r2) + (301056*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 0.010416666666666666
        tmp14 = tmp12 * tmp13
        tmp17 = tmp15 * tmp16
        tmp18 = 96.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp19 - tmp4
        tmp22 = tmp21 * tmp9
        tmp23 = tmp20 - tmp22
        tmp24 = tmp14 * tmp23
        tmp25 = tmp11 + tmp24
        tmp27 = tmp25 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
        tmp33 = tmp31 - tmp32
        tmp35 = tmp33 * tmp34
        tmp36 = tmp27 * tmp35
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask & xmask, tmp39, _tmp38)
        tl.store(in_out_ptr0 + (r2 + (96*x3)), tmp25, rmask & xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp42 = tl.load(in_out_ptr0 + (r2 + (96*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr5 + (x0 + (3136*r2) + (301056*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = 0.010416666666666666
        tmp41 = tmp34 * tmp40
        tmp44 = tmp42 * tmp43
        tmp45 = 96.0
        tmp46 = tmp44 * tmp45
        tmp47 = tmp46 - tmp29
        tmp49 = tmp48 - tmp32
        tmp50 = tmp49 * tmp34
        tmp51 = tmp50 * tmp38
        tmp52 = tmp47 - tmp51
        tmp53 = tmp41 * tmp52
        tl.store(out_ptr4 + (r2 + (96*x3)), tmp53, rmask & xmask)
