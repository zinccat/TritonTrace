# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_32', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr3, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    r3 = rindex
    x2 = (xindex // 196)
    x6 = xindex
    x4 = xindex % 196
    tmp48 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr2 + (r3 + (384*x6)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((-1) + x1) % 2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.broadcast_to(x0, [RBLOCK])
    tmp8 = tmp7 >= tmp1
    tmp9 = tl.broadcast_to(((-1) + x0) % 2, [RBLOCK])
    tmp10 = tmp9 == tmp4
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11 & tmp6
    tmp13 = tl.load(in_ptr0 + (1152 + r3 + (1536*(triton_helpers.div_floor_integer((-1) + x0,  2))) + (10752*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (75264*x2)), rmask & tmp12, other=0.0)
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp6, tmp17, tmp14)
    tmp19 = ((x6 // 14) % 14) % 2
    tmp20 = tmp19 == tmp4
    tmp21 = tmp11 & tmp20
    tmp22 = tl.load(in_ptr0 + (768 + r3 + (1536*(triton_helpers.div_floor_integer((-1) + x0,  2))) + (10752*(x1 // 2)) + (75264*x2)), rmask & tmp21, other=0.0)
    tmp23 = tl.where(tmp11, tmp22, tmp14)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = tl.where(tmp20, tmp25, tmp14)
    tmp27 = tmp18 + tmp26
    tmp28 = tl.broadcast_to(x6 % 2, [RBLOCK])
    tmp29 = tmp28 == tmp4
    tmp30 = tmp29 & tmp6
    tmp31 = tl.load(in_ptr0 + (384 + r3 + (1536*(x0 // 2)) + (10752*(triton_helpers.div_floor_integer((-1) + x1,  2))) + (75264*x2)), rmask & tmp30, other=0.0)
    tmp32 = tl.where(tmp29, tmp31, tmp14)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tl.where(tmp6, tmp34, tmp14)
    tmp36 = tmp27 + tmp35
    tmp37 = ((x6 % 196) // 14) % 2
    tmp38 = tmp37 == tmp4
    tmp39 = tl.broadcast_to((x4 % 14) % 2, [RBLOCK])
    tmp40 = tmp39 == tmp4
    tmp41 = tmp40 & tmp38
    tmp42 = tl.load(in_ptr0 + (r3 + (1536*((x4 % 14) // 2)) + (10752*(x4 // 28)) + (75264*x2)), rmask & tmp41, other=0.0)
    tmp43 = tl.where(tmp40, tmp42, tmp14)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp38, tmp43, tmp44)
    tmp46 = tl.where(tmp38, tmp45, tmp14)
    tmp47 = tmp36 + tmp46
    tmp49 = tmp47 * tmp48
    tmp50 = tl.broadcast_to(tmp49, [RBLOCK])
    tmp52 = tl.where(rmask, tmp50, 0)
    tmp53 = triton_helpers.promote_to_tensor(tl.sum(tmp52, 0))
    tmp56 = tmp54 - tmp55
    tmp58 = tmp56 * tmp57
    tmp59 = tmp49 * tmp58
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp64 = 0.0026041666666666665
    tmp65 = tmp57 * tmp64
    tmp66 = 384.0
    tmp67 = tmp49 * tmp66
    tmp68 = tmp67 - tmp53
    tmp69 = tmp58 * tmp63
    tmp70 = tmp68 - tmp69
    tmp71 = tmp65 * tmp70
    tl.store(out_ptr0 + (r3 + (384*x6)), tmp36, rmask)
    tl.store(out_ptr3 + (r3 + (384*x6)), tmp71, rmask)
