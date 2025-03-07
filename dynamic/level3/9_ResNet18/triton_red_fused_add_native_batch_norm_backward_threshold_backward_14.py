# From: 9_ResNet18

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex % ks0
        r4 = (rindex // ks0)
        tmp0 = tl.load(in_ptr0 + (r3 + x0 + (128*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (128*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (256*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r3 + x0 + (128*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (128*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (256*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + x0 + (128*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (128*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (256*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r3 + x0 + (128*r4) + (x0*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  8))) + (128*r4*((triton_helpers.div_floor_integer((-1) + ks1,  8))*(triton_helpers.div_floor_integer((-1) + ks1,  8)))) + (256*r4*(triton_helpers.div_floor_integer((-1) + ks1,  8)))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
