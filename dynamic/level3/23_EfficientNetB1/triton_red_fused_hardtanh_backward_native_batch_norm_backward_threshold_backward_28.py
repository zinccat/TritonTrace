# From: 23_EfficientNetB1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex % ks0
        r4 = (rindex // ks0)
        tmp0 = tl.load(in_ptr0 + (r3 + x0 + (240*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (240*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (480*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r3 + x0 + (240*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (240*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (480*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r3 + x0 + (240*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (240*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (480*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp3 = 6.0
        tmp4 = tmp0 >= tmp3
        tmp5 = tmp2 | tmp4
        tmp7 = tl.where(tmp5, tmp1, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp7 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp16, xmask)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
