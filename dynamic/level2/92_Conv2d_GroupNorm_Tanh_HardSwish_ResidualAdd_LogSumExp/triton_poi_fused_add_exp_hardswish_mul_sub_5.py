# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_hardswish_mul_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_hardswish_mul_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex % ks0)
    x5 = xindex // ks1
    x6 = xindex
    x7 = xindex // ks0
    x1 = ((xindex // ks3) % 16)
    tmp0 = tl.load(in_ptr0 + (x3 + 4*x5 + x5*ks2*ks2 + ((-4)*ks2*x5)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x6), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x3 + 4*x5 + x5*ks2*ks2 + ((-4)*ks2*x5)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp1 + tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp0 * tmp15
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 * tmp20
    tmp23 = tmp1 * tmp22
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tmp16 + tmp26
    tl.store(in_out_ptr0 + (x6), tmp27, xmask)
