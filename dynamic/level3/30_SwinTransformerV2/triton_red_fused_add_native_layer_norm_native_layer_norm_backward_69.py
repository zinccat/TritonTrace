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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32', 22: 'i32', 23: 'i32', 24: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_69', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 13, 'num_reduction': 6, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp58 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp63 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((15 + (196*ks0)) // 16))
        tmp1 = 196*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp9 = tl.load(in_ptr4 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tmp17 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp18 = tl.where(tmp2, tmp5, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr5 + (x0 + (384*((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14) % 7)) + (2688*(((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14) % 7)) + (18816*(((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % 14) // 7) % 2)) + (37632*((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // 14) % 14) // 7) % 2)) + (75264*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp5 + tmp22
        tmp24 = tl.load(in_ptr6 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr7 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp24 - tmp25
        tmp27 = tl.load(in_ptr8 + ((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 * tmp27
        tmp29 = tmp23 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
        tmp35 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp36 = tl.where(tmp2, tmp23, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp40 = tl.load(in_ptr9 + (x0 + (384*((r2 + (x1*((15 + (196*ks0)) // 16))) % (196*ks0)))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp41 = tmp23 + tmp40
        tl.device_assert(((0 <= tl.where((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 0, 14 + (((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)), ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1))) & (tl.where((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 0, 14 + (((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)), ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 14)) | ~(rmask & tmp2), "index out of bounds: 0 <= tl.where((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 0, 14 + (((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)), ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 14")
        tmp43 = tl.load(in_ptr10 + ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) // (7*ks1)) % 14), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.full([XBLOCK, RBLOCK], 14, tl.int32)
        tmp45 = tmp43 + tmp44
        tmp46 = tmp43 < 0
        tmp47 = tl.where(tmp46, tmp45, tmp43)
        tl.device_assert(((0 <= tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])) & (tl.broadcast_to(tmp47, [XBLOCK, RBLOCK]) < 14)) | ~(rmask & tmp2), "index out of bounds: 0 <= tl.broadcast_to(tmp47, [XBLOCK, RBLOCK]) < 14")
        tmp49 = tl.load(in_ptr11 + (x0 + (384*((tl.where((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 0, 14 + (((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)), ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1))) % 7)) + (2688*(tmp47 % 7)) + (18816*(((tl.where((((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)) < 0, 14 + (((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1)), ((((r2 + (x1*((15 + (196*ks0)) // 16))) % 196) % (7*ks1)) + (triton_helpers.remainder_integer((-3) + (7*ks1), 7*ks1))) % (7*ks1))) // 7) % ks1)) + (18816*ks1*((tmp47 // 7) % 2)) + (18816*(triton_helpers.div_floor_integer(ks2,  libdevice.trunc((ks2.to(tl.float64)) / 4.00000000000000).to(tl.int32)))*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0))), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp50 = tl.load(in_ptr12 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp51 = tmp49 - tmp50
        tmp52 = tl.load(in_ptr13 + ((98*ks1*(((r2 + (x1*((15 + (196*ks0)) // 16))) // 196) % ks0)) + ((r2 + (x1*((15 + (196*ks0)) // 16))) % 196)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp53 = tmp51 * tmp52
        tmp54 = tmp41 * tmp53
        tmp55 = tl.full(tmp54.shape, 0, tmp54.dtype)
        tmp56 = tl.where(tmp2, tmp54, tmp55)
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
        tmp59 = _tmp58 + tmp57
        _tmp58 = tl.where(rmask, tmp59, _tmp58)
        tmp60 = tl.full(tmp41.shape, 0, tmp41.dtype)
        tmp61 = tl.where(tmp2, tmp41, tmp60)
        tmp62 = tl.broadcast_to(tmp61, [XBLOCK, RBLOCK])
        tmp64 = _tmp63 + tmp62
        _tmp63 = tl.where(rmask, tmp64, _tmp63)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, None)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp33, None)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp38, None)
    tmp58 = tl.sum(_tmp58, 1)[:, None]
    tl.store(out_ptr4 + (x3), tmp58, None)
    tmp63 = tl.sum(_tmp63, 1)[:, None]
    tl.store(out_ptr5 + (x3), tmp63, None)
