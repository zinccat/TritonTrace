# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x7 = xindex // ks0
    x9 = xindex // ks4
    x10 = ((xindex // ks0) % 16)
    x11 = xindex
    tmp0 = tl.load(in_ptr0 + (((-1)*x7) + ((-1)*(((x0 // ks1) % ks1))) + ((-4)*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3))) + ((-4)*x7*ks3*ks3) + 2*ks2*x7 + 2*ks3*(((x0 // ks1) % ks1)) + 4*ks3*x7 + 4*ks3*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((-8)*ks2*ks3*x7) + 8*ks2*x7*ks3*ks3 + (triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (((-1)*x7) + ((-1)*(((x0 // ks1) % ks1))) + ((-4)*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3))) + ((-4)*x7*ks3*ks3) + 2*ks2*x7 + 2*ks3*(((x0 // ks1) % ks1)) + 4*ks3*x7 + 4*ks3*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((-8)*ks2*ks3*x7) + 8*ks2*x7*ks3*ks3 + (triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x9), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x10), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (((-1)*x7) + ((-1)*(((x0 // ks1) % ks1))) + ((-4)*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3))) + ((-4)*x7*ks3*ks3) + 2*ks2*x7 + 2*ks3*(((x0 // ks1) % ks1)) + 4*ks3*x7 + 4*ks3*ks3*(triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((-8)*ks2*ks3*x7) + 8*ks2*x7*ks3*ks3 + (triton_helpers.div_floor_integer(x0,  1 + ((-4)*ks3) + 4*ks3*ks3)) + ((x0 % ks1))), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x9), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (x7 // 4), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr4 + (x11), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = 0.3333333333333333
    tmp7 = tmp0 * tmp6
    tmp8 = 0.5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp5 * tmp9
    tmp11 = tl.where(tmp4, tmp10, tmp5)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp12, tmp11)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp19 * tmp18
    tmp22 = tmp21 * tmp14
    tmp23 = tmp22 * tmp14
    tmp24 = tmp23 * tmp14
    tmp25 = 2.0
    tmp26 = ks3
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp29 = -1.0
    tmp30 = tmp29 + tmp28
    tmp31 = libdevice.pow(tmp30, tmp25)
    tmp32 = 4.0
    tmp33 = tmp32 * tmp31
    tmp34 = ks2
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp25 * tmp35
    tmp37 = tmp29 + tmp36
    tmp38 = tmp33 * tmp37
    tmp39 = tmp38.to(tl.float64)
    tmp40 = tl.full([1], 1.0, tl.float64)
    tmp41 = tmp40 / tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp24 * tmp42
    tmp44 = tmp20 * tmp43
    tmp45 = tmp17 + tmp44
    tmp47 = tmp45 + tmp46
    tmp49 = tl.sigmoid(tmp48)
    tmp50 = tmp47 * tmp49
    tmp51 = tmp47 * tmp48
    tmp52 = 1.0
    tmp53 = tmp52 - tmp49
    tmp54 = tmp49 * tmp53
    tmp55 = tmp51 * tmp54
    tmp56 = tmp50 + tmp55
    tl.store(in_out_ptr0 + (x11), tmp56, xmask)
