# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_cumsum_exp_flip_mul_neg_select_backward_sub_sum_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % ks0
    x1 = (xindex // ks0) % 8
    x2 = (xindex // ks1)
    x4 = xindex
    tmp1 = tl.load(in_ptr1 + ((-1) + ks2 + (ks2*x4)), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (ks2*x1) + (8*ks2*x0) + (8*ks0*ks2*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + (ks2*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tl_math.exp(tmp3)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    x5 = (xindex // ks0)
    tmp30 = tl.full([XBLOCK, 1], float('nan'), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp18 = tl.load(in_ptr0 + ((-1) + ks2 + ((-1)*r3) + (ks2*x1) + (8*ks2*x0) + (8*ks0*ks2*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr1 + ((-1) + ks2 + ((-1)*r3) + (ks2*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (-1) + ks2 + ((-1)*r3)
        tmp10 = (-1) + ks2
        tmp11 = tmp9 == tmp10
        tmp12 = 1 + x0
        tmp13 = tl.full([1, 1], 0, tl.int64)
        tmp14 = tmp12 >= tmp13
        tmp15 = tl.load(in_ptr2 + (tl.broadcast_to((-1) + ks0 + x5 + ((-1)*x0) + (ks0*x5), [XBLOCK, RBLOCK])), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 0.0
        tmp17 = tl.where(tmp11, tmp15, tmp16)
        tmp20 = tmp1 - tmp19
        tmp21 = tl_math.exp(tmp20)
        tmp22 = tmp18 * tmp21
        tmp23 = -tmp22
        tmp24 = tmp17 + tmp23
        tmp25 = tmp9 >= tmp10
        tmp26 = tl.where(tmp25, tmp7, tmp16)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31, = tl.associative_scan((tmp29,), 1, _triton_helper_fn_add0)
        tmp32 = triton_helpers.select_one((tmp31), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp33 = tmp30 + tmp32
        tmp34 = tmp30 + tmp31
        tmp35 = tl.where(roffset > 0, tmp34, tmp31)
        tmp30 = tl.where(roffset > 0, tmp33, tmp32)
        tl.store(out_ptr1 + (r3 + (ks2*x4)), tmp35, rmask & xmask)
