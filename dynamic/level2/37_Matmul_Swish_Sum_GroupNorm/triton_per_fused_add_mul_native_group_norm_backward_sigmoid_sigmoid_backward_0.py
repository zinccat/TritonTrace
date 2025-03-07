# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r2 + 32*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 32*x3), xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2 + 32*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + 32*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp0 * tmp7
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp18 * tmp7
    tmp20 = tmp0 * tmp19
    tmp22 = tmp17 * tmp21
    tmp23 = tmp22 - tmp12
    tmp24 = tmp23 * tmp18
    tmp25 = tmp24 * tmp18
    tmp26 = tmp25 * tmp18
    tmp27 = 0.03125
    tmp28 = tmp26 * tmp27
    tmp29 = tmp5 * tmp28
    tmp30 = tmp20 + tmp29
    tmp31 = -tmp28
    tmp32 = tmp31 * tmp21
    tmp33 = tmp17 * tmp18
    tmp34 = tmp33 * tmp27
    tmp35 = tmp32 - tmp34
    tmp36 = tmp30 + tmp35
    tmp37 = tmp36 * tmp2
    tmp38 = tmp36 * tmp1
    tmp39 = 1.0
    tmp40 = tmp39 - tmp2
    tmp41 = tmp2 * tmp40
    tmp42 = tmp38 * tmp41
    tmp43 = tmp37 + tmp42
    tl.store(out_ptr2 + (r2 + 32*x3), tmp36, xmask)
    tl.store(out_ptr3 + (r2 + 32*x3), tmp43, xmask)
