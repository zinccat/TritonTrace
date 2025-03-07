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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_70', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (384*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (384*((x0 % 14) % 7)) + (2688*((x0 // 14) % 7)) + (18816*((x0 % 14) // 7)) + (37632*(x0 // 98)) + (75264*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tl.device_assert((0 <= tl.where((((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) & (tl.where((((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 14), "index out of bounds: 0 <= tl.where((((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 14")
    tmp14 = tl.load(in_ptr5 + (((x3 % 196) // (7*ks0)) % 14), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0 + (98*ks0*x1)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0 + (98*ks0*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tl.full([RBLOCK], 14, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert((0 <= tmp18) & (tmp18 < 14), "index out of bounds: 0 <= tmp18 < 14")
    tmp20 = tl.load(in_ptr6 + (r2 + (384*((tl.where((((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) % 7)) + (2688*(tmp18 % 7)) + (18816*(((tl.where((((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)) < 0, 14 + (((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0)), ((x0 % (7*ks0)) + (triton_helpers.remainder_integer((-3) + (7*ks0), 7*ks0))) % (7*ks0))) // 7) % ks0)) + (18816*ks0*((tmp18 // 7) % 2)) + (18816*x1*(triton_helpers.div_floor_integer(ks1,  libdevice.trunc((ks1.to(tl.float64)) / 4.00000000000000).to(tl.int32))))), rmask, other=0.0)
    tmp22 = tmp20 - tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp8 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp8, rmask)
    tl.store(out_ptr1 + (x3), tmp12, None)
    tl.store(out_ptr2 + (x3), tmp29, None)
