# From: 9_ResNet18

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32', 18: 'i32', 19: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex % ks0
        r4 = (rindex // ks0)
        tmp0 = tl.load(in_ptr0 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr6 + (r3 + x0 + (512*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))) + (512*r4*((triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)))) + (1024*r4*(triton_helpers.div_floor_integer((-1) + ks1,  32)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = ks0
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 / tmp6
        tmp8 = tl.where(tmp3, tmp1, tmp7)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp11 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp27, xmask)
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tmp20 * tmp29
    tmp32 = tmp27 * tmp31
    tl.store(out_ptr4 + (x0), tmp30, xmask)
    tl.store(out_ptr5 + (x0), tmp32, xmask)
