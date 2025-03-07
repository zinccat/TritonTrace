# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_leaky_relu_leaky_relu_backward_native_group_norm_native_group_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_leaky_relu_leaky_relu_backward_native_group_norm_native_group_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (r2 + 32*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (r2 + 32*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r2 + 32*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr5 + (r2 + 32*x3), xmask, other=0.0)
    tmp27 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp12 = tmp11 + tmp11
    tmp13 = 0.01
    tmp14 = tmp12 * tmp13
    tmp15 = tl.where(tmp10, tmp12, tmp14)
    tmp16 = tmp15 * tmp0
    tmp17 = tmp16 * tmp5
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp15 * tmp5
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp28 = tmp27 * tmp5
    tmp29 = tmp15 * tmp28
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 - tmp21
    tmp33 = tmp32 * tmp27
    tmp34 = tmp33 * tmp27
    tmp35 = tmp34 * tmp27
    tmp36 = 0.03125
    tmp37 = tmp35 * tmp36
    tmp38 = tmp0 * tmp37
    tmp39 = tmp29 + tmp38
    tmp40 = -tmp37
    tmp41 = tmp40 * tmp30
    tmp42 = tmp26 * tmp27
    tmp43 = tmp42 * tmp36
    tmp44 = tmp41 - tmp43
    tmp45 = tmp39 + tmp44
    tl.store(out_ptr0 + (r2 + 32*x3), tmp15, xmask)
    tl.store(out_ptr3 + (r2 + 32*x3), tmp45, xmask)
