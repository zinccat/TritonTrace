# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_mul_sum_100', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 282
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 6)
    x0 = xindex % 6
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((46 + (38416*ks0)) // 47))
        tmp1 = 38416*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((2401*x0) + (14406*(((r2 + (x1*((46 + (38416*ks0)) // 47))) // 2401) % ks1)) + ((r2 + (x1*((46 + (38416*ks0)) // 47))) % 2401)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -tmp3
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (294*(((r2 + (x1*((46 + (38416*ks0)) // 47))) // 2401) % ks1)) + (((r2 + (x1*((46 + (38416*ks0)) // 47))) // 49) % 49)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((2401*x0) + (14406*(((r2 + (x1*((46 + (38416*ks0)) // 47))) // 2401) % ks1)) + ((r2 + (x1*((46 + (38416*ks0)) // 47))) % 2401)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 * tmp3
        tmp8 = libdevice.fma(tmp4, tmp5, tmp7)
        tmp9 = tl.load(in_ptr3 + ((2401*x0) + (14406*(((r2 + (x1*((46 + (38416*ks0)) // 47))) // 2401) % ks1)) + ((r2 + (x1*((46 + (38416*ks0)) // 47))) % 2401)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
