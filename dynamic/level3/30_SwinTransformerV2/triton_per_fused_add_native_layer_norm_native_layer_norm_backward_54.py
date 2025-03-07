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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (r2 + (384*x3)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + ((x0 // 14)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x3 % 14), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr5 + (r2 + (384*x3)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp1 = ((x3 % 196) // 14) % 2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.broadcast_to((x0 % 14) % 2, [RBLOCK])
    tmp5 = tmp4 == tmp2
    tmp6 = tmp5 & tmp3
    tmp7 = tl.load(in_ptr0 + (r2 + (1536*((x0 % 14) // 2)) + (10752*(x0 // 28)) + (75264*x1)), rmask & tmp6, other=0.0)
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp3, tmp9, tmp10)
    tmp12 = tl.where(tmp3, tmp11, tmp8)
    tmp13 = tmp0 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tl.full([RBLOCK], 14, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tl.device_assert((0 <= tmp20) & (tmp20 < 14), "index out of bounds: 0 <= tmp20 < 14")
    tmp23 = tmp22 + tmp17
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tl.device_assert((0 <= tmp25) & (tmp25 < 14), "index out of bounds: 0 <= tmp25 < 14")
    tmp27 = tl.load(in_ptr3 + (r2 + (384*(tmp25 % 7)) + (2688*(tmp20 % 7)) + (18816*((tmp25 // 7) % 2)) + (37632*((tmp20 // 7) % 2)) + (75264*x1)), rmask, other=0.0)
    tmp28 = tmp15 + tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp37 = tmp35 - tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tmp30 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp45 = 0.0026041666666666665
    tmp46 = tmp38 * tmp45
    tmp47 = 384.0
    tmp48 = tmp30 * tmp47
    tmp49 = tmp48 - tmp34
    tmp50 = tmp39 * tmp44
    tmp51 = tmp49 - tmp50
    tmp52 = tmp46 * tmp51
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp28, rmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp52, rmask)
