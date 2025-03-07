# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32', 20: 'i32', 21: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_72', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 6, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp47 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp52 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((15 + (196*ks0)) // 16))
        tmp1 = 196*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr4 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp3 + tmp18
        tmp20 = tl.load(in_ptr5 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr6 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 - tmp21
        tmp23 = tl.load(in_ptr7 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 * tmp23
        tmp25 = tmp19 * tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
        tmp31 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp32 = tl.where(tmp2, tmp19, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tmp36 = tl.load(in_ptr8 + (x0 + (384*((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14) % 7)) + (2688*(((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14) % 7)) + (18816*(((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14) // 7) % 2)) + (37632*((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14) // 7) % 2)) + (75264*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp37 = tmp19 + tmp36
        tmp38 = tl.load(in_ptr9 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr10 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp40 = tmp38 - tmp39
        tmp41 = tl.load(in_ptr11 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp42 = tmp40 * tmp41
        tmp43 = tmp37 * tmp42
        tmp44 = tl.full(tmp43.shape, 0, tmp43.dtype)
        tmp45 = tl.where(tmp2, tmp43, tmp44)
        tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
        tmp48 = _tmp47 + tmp46
        _tmp47 = tl.where(rmask, tmp48, _tmp47)
        tmp49 = tl.full(tmp37.shape, 0, tmp37.dtype)
        tmp50 = tl.where(tmp2, tmp37, tmp49)
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(rmask, tmp53, _tmp52)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, None)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp29, None)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp34, None)
    tmp47 = tl.sum(_tmp47, 1)[:, None]
    tl.store(out_ptr4 + (x3), tmp47, None)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tl.store(out_ptr5 + (x3), tmp52, None)
