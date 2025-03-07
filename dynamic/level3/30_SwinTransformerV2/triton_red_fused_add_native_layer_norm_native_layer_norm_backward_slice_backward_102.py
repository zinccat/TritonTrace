# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_102', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136)
    x6 = xindex
    x4 = xindex % 3136
    _tmp51 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp54 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp48 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp53 = tl.load(in_ptr2 + (r3 + (96*x6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ((-1) + x1) % 2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 == tmp4
        tmp6 = tmp2 & tmp5
        tmp7 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
        tmp8 = tmp7 >= tmp1
        tmp9 = tl.broadcast_to(((-1) + x0) % 2, [XBLOCK, RBLOCK])
        tmp10 = tmp9 == tmp4
        tmp11 = tmp8 & tmp10
        tmp12 = tmp11 & tmp6
        tmp13 = tl.load(in_ptr0 + (288 + r3 + (384*(triton_helpers.div_floor_integer((-1) + x0,  2))) + (10752*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (301056*x2)), rmask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = 0.0
        tmp15 = tl.where(tmp11, tmp13, tmp14)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp6, tmp15, tmp16)
        tmp18 = tl.where(tmp6, tmp17, tmp14)
        tmp19 = ((x6 // 56) % 56) % 2
        tmp20 = tmp19 == tmp4
        tmp21 = tmp11 & tmp20
        tmp22 = tl.load(in_ptr0 + (192 + r3 + (384*(triton_helpers.div_floor_integer((-1) + x0,  2))) + (10752*(x1 // 2)) + (301056*x2)), rmask & tmp21 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.where(tmp11, tmp22, tmp14)
        tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
        tmp25 = tl.where(tmp20, tmp23, tmp24)
        tmp26 = tl.where(tmp20, tmp25, tmp14)
        tmp27 = tmp18 + tmp26
        tmp28 = tl.broadcast_to(x6 % 2, [XBLOCK, RBLOCK])
        tmp29 = tmp28 == tmp4
        tmp30 = tmp29 & tmp6
        tmp31 = tl.load(in_ptr0 + (96 + r3 + (384*(x0 // 2)) + (10752*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (301056*x2)), rmask & tmp30 & xmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.where(tmp29, tmp31, tmp14)
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp6, tmp32, tmp33)
        tmp35 = tl.where(tmp6, tmp34, tmp14)
        tmp36 = tmp27 + tmp35
        tmp37 = ((x6 % 3136) // 56) % 2
        tmp38 = tmp37 == tmp4
        tmp39 = tl.broadcast_to((x4 % 56) % 2, [XBLOCK, RBLOCK])
        tmp40 = tmp39 == tmp4
        tmp41 = tmp40 & tmp38
        tmp42 = tl.load(in_ptr0 + (r3 + (384*((x4 % 56) // 2)) + (10752*(x4 // 112)) + (301056*x2)), rmask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.where(tmp40, tmp42, tmp14)
        tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
        tmp45 = tl.where(tmp38, tmp43, tmp44)
        tmp46 = tl.where(tmp38, tmp45, tmp14)
        tmp47 = tmp36 + tmp46
        tmp49 = tmp47 * tmp48
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
        tmp52 = _tmp51 + tmp50
        _tmp51 = tl.where(rmask & xmask, tmp52, _tmp51)
        tmp55 = tmp53 - tmp54
        tmp57 = tmp55 * tmp56
        tmp58 = tmp49 * tmp57
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask & xmask, tmp61, _tmp60)
        tl.store(out_ptr0 + (r3 + (96*x6)), tmp36, rmask & xmask)
    tmp51 = tl.sum(_tmp51, 1)[:, None]
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp64 = tl.load(out_ptr0 + (r3 + (96*x6)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp78 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.load(in_ptr2 + (r3 + (96*x6)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp62 = 0.010416666666666666
        tmp63 = tmp56 * tmp62
        tmp65 = ((x6 % 3136) // 56) % 2
        tmp66 = tl.full([1, 1], 0, tl.int64)
        tmp67 = tmp65 == tmp66
        tmp68 = tl.broadcast_to((x4 % 56) % 2, [XBLOCK, RBLOCK])
        tmp69 = tmp68 == tmp66
        tmp70 = tmp69 & tmp67
        tmp71 = tl.load(in_ptr0 + (r3 + (384*((x4 % 56) // 2)) + (10752*(x4 // 112)) + (301056*x2)), rmask & tmp70 & xmask, eviction_policy='evict_first', other=0.0)
        tmp72 = 0.0
        tmp73 = tl.where(tmp69, tmp71, tmp72)
        tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
        tmp75 = tl.where(tmp67, tmp73, tmp74)
        tmp76 = tl.where(tmp67, tmp75, tmp72)
        tmp77 = tmp64 + tmp76
        tmp79 = tmp77 * tmp78
        tmp80 = 96.0
        tmp81 = tmp79 * tmp80
        tmp82 = tmp81 - tmp51
        tmp84 = tmp83 - tmp54
        tmp85 = tmp84 * tmp56
        tmp86 = tmp85 * tmp60
        tmp87 = tmp82 - tmp86
        tmp88 = tmp63 * tmp87
        tl.store(out_ptr3 + (r3 + (96*x6)), tmp88, rmask & xmask)
