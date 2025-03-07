# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_native_batch_norm_backward_slice_backward_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((24 + (ks0*ks1)) // 25))
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*r2) + (48*x1*((24 + (ks0*ks1)) // 25))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -tmp3
        tmp5 = tl.load(in_ptr1 + (r2 + (x1*((24 + (ks0*ks1)) // 25))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
        tmp7 = tl.full([1, 1], 32, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp2
        tmp10 = tl.load(in_ptr2 + ((ks1*x0) + (32*ks1*(((r2 + (x1*((24 + (ks0*ks1)) // 25))) // ks1) % ks0)) + ((r2 + (x1*((24 + (ks0*ks1)) // 25))) % ks1)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (32*(((r2 + (x1*((24 + (ks0*ks1)) // 25))) // ks1) % ks0))), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = 0.0
        tmp16 = tl.where(tmp8, tmp14, tmp15)
        tmp17 = tmp16 * tmp3
        tmp18 = libdevice.fma(tmp4, tmp5, tmp17)
        tmp19 = tl.load(in_ptr4 + (x0 + (48*r2) + (48*x1*((24 + (ks0*ks1)) // 25))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
