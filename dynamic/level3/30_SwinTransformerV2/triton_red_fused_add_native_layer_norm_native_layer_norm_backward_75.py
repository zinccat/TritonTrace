# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_75', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp46 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((61 + (784*ks0)) // 62))
        tmp1 = 784*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*(((r2 + (x1*((61 + (784*ks0)) // 62))) % 784) % 784)) + (150528*(((r2 + (x1*((61 + (784*ks0)) // 62))) // 784) % ks0))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = ((((r2 + (x1*((61 + (784*ks0)) // 62))) % 784) // 28) % 28) % 2
        tmp5 = tl.full([1, 1], 0, tl.int64)
        tmp6 = tmp4 == tmp5
        tmp7 = tmp6 & tmp2
        tmp8 = (((r2 + (x1*((61 + (784*ks0)) // 62))) % 784) % 28) % 2
        tmp9 = tmp8 == tmp5
        tmp10 = tmp9 & tmp7
        tmp11 = tl.load(in_ptr1 + (x0 + (768*(triton_helpers.div_floor_integer(((r2 + (x1*((61 + (784*ks0)) // 62))) % 784) % 28,  2))) + (10752*(triton_helpers.div_floor_integer((((r2 + (x1*((61 + (784*ks0)) // 62))) % 784) // 28) % 28,  2))) + (150528*(((r2 + (x1*((61 + (784*ks0)) // 62))) // 784) % ks0))), rmask & tmp10 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = 0.0
        tmp13 = tl.where(tmp9, tmp11, tmp12)
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp7, tmp13, tmp14)
        tmp16 = tl.where(tmp6, tmp15, tmp12)
        tmp17 = tmp3 + tmp16
        tmp18 = tl.load(in_ptr2 + (x0 + (192*((r2 + (x1*((61 + (784*ks0)) // 62))) % (784*ks0)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + ((r2 + (x1*((61 + (784*ks0)) // 62))) % (784*ks0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 - tmp19
        tmp21 = tl.load(in_ptr4 + ((r2 + (x1*((61 + (784*ks0)) // 62))) % (784*ks0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tmp17 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp29 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp30 = tl.where(tmp2, tmp17, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
        tmp34 = tl.load(in_ptr5 + (x0 + (192*((r2 + (x1*((61 + (784*ks0)) // 62))) % (784*ks0)))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tmp17 + tmp34
        tmp36 = tl.load(in_ptr6 + (x0 + (192*((r2 + (x1*((61 + (784*ks0)) // 62))) % 784)) + (37632*(triton_helpers.div_floor_integer(ks1,  libdevice.trunc(((16*ks0).to(tl.float64)) / 16.0000000000000).to(tl.int32)))*(((r2 + (x1*((61 + (784*ks0)) // 62))) // 784) % ks0))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tmp35 * tmp36
        tmp38 = tl.full(tmp37.shape, 0, tmp37.dtype)
        tmp39 = tl.where(tmp2, tmp37, tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tmp43 = tl.full(tmp35.shape, 0, tmp35.dtype)
        tmp44 = tl.where(tmp2, tmp35, tmp43)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(rmask & xmask, tmp47, _tmp46)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp41, xmask)
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp46, xmask)
