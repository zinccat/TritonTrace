# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16777216, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (r1 + 64*x0), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7071067811865476
    tmp13 = tmp8 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = tmp14 + tmp10
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp8 * tmp8
    tmp19 = -0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = 0.3989422804014327
    tmp23 = tmp21 * tmp22
    tmp24 = tmp8 * tmp23
    tmp25 = tmp17 + tmp24
    tmp26 = tmp11 * tmp25
    tmp27 = tmp26 * tmp5
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tmp27 * tmp4
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.sum(tmp32, 1)[:, None]
    tmp35 = 64.0
    tmp36 = tmp27 * tmp35
    tmp37 = tmp36 - tmp30
    tmp38 = tmp4 * tmp34
    tmp39 = tmp37 - tmp38
    tmp40 = 0.015625
    tmp41 = tmp3 * tmp40
    tmp42 = tmp41 * tmp39
    tl.store(out_ptr0 + (r1 + 64*x0), tmp8, None)
    tl.store(in_out_ptr0 + (r1 + 64*x0), tmp42, None)
