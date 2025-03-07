# From: 24_EfficientNetB2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(rmask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask & xmask, tmp11_weight_next, tmp11_weight)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp15 = tl.load(in_ptr1 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (x0 + (864*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = 1.0
        tmp17 = tmp15 * tmp16
        tmp18 = 0.0
        tmp19 = tl.where(tmp14, tmp18, tmp17)
        tmp21 = tmp20 - tmp11
        tmp22 = tmp19 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(rmask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(rmask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(rmask & xmask, tmp27_weight_next, tmp27_weight)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tmp27_tmp, tmp28_tmp, tmp29_tmp = triton_helpers.welford(
        tmp27_mean, tmp27_m2, tmp27_weight, 1
    )
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    tmp29 = tmp29_tmp[:, None]
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tmp30 = ks0
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp24 * tmp35
    tl.store(out_ptr4 + (x0), tmp36, xmask)
