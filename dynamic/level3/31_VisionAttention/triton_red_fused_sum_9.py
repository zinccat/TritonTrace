# From: 31_VisionAttention

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + ((32*((((128*(r2 % ks0)) + (128*ks0*(((r2 + (64*ks0*x1)) // ks0) % 16384)) + (x0 % 128)) // (128*ks0)) % 16384)) + (524288*((((128*(r2 % ks0)) + (x0 % 128)) // 32) % (4*ks0))) + ((x0 % 128) % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((16384*((x0 % 128) % 32)) + (524288*((((128*(r2 % ks0)) + (x0 % 128)) // 32) % (4*ks0))) + ((((128*(r2 % ks0)) + (128*ks0*(((r2 + (64*ks0*x1)) // ks0) % 16384)) + (x0 % 128)) // (128*ks0)) % 16384)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + ((32*((((128*(r2 % ks0)) + (128*ks0*(((r2 + (64*ks0*x1)) // ks0) % 16384)) + (x0 % 128)) // (128*ks0)) % 16384)) + (524288*((((128*(r2 % ks0)) + (x0 % 128)) // 32) % (4*ks0))) + ((x0 % 128) % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = (x0 // 128)
        tmp1 = tl.full([1, 1], 2, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 1, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.full([1, 1], 0, tl.int32)
        tmp12 = tmp0 == tmp11
        tmp14 = 0.1767766952966369
        tmp15 = tmp13 * tmp14
        tmp16 = tl.where(tmp12, tmp15, tmp4)
        tmp17 = tmp10 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, None)
