# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool3d_backward_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_avg_pool3d_backward_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % ks0)
    x2 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))))) + 1024*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2))))) + ((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) * (((-1) + ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 1024*ks2*x2 + ((((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))) * ((((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))) <= ((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32))))) + ((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))) * (((-1) + ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))) < (((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0))))))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (r3 + 64*x4), None)
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp1 = tmp0 / 8
    tmp2 = ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp3 = ((ks2) * ((ks2) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks2)))
    tmp4 = tmp2 < tmp3
    tmp5 = ((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))
    tmp6 = ((32) * ((32) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (32)))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = ((0) * ((0) >= (r3 // 2)) + (r3 // 2) * ((r3 // 2) > (0)))
    tmp10 = ((32) * ((32) <= (1 + (r3 // 2))) + (1 + (r3 // 2)) * ((1 + (r3 // 2)) < (32)))
    tmp11 = tmp9 < tmp10
    tmp12 = tmp8 & tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp12, tmp1, tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp23 = tmp20 + tmp22
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tmp16 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = 0.015625
    tmp33 = tmp26 * tmp32
    tmp34 = 64.0
    tmp35 = tmp16 * tmp34
    tmp36 = tmp35 - tmp19
    tmp37 = tmp27 * tmp31
    tmp38 = tmp36 - tmp37
    tmp39 = tmp33 * tmp38
    tl.store(out_ptr0 + (r3 + 64*x4), tmp14, None)
    tl.store(out_ptr3 + (r3 + 64*x4), tmp39, None)
