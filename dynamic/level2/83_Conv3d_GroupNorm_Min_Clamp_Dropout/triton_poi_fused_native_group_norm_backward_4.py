# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_backward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x4 = xindex // ks0
    x6 = xindex // ks3
    x7 = ((xindex // ks0) % 16)
    x8 = xindex
    tmp0 = tl.load(in_ptr0 + (((-8)*x4) + ((-2)*(((x0 // ((-2) + ks2)) % ((-2) + ks2)))) + 4*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ks2*(((x0 // ((-2) + ks2)) % ((-2) + ks2))) + ks2*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ((-4)*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2)))) + ((-2)*x4*ks2*ks2) + 4*ks1*x4 + 8*ks2*x4 + ks1*x4*ks2*ks2 + ((-4)*ks1*ks2*x4) + ((x0 % ((-2) + ks2)))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (((-8)*x4) + ((-2)*(((x0 // ((-2) + ks2)) % ((-2) + ks2)))) + 4*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ks2*(((x0 // ((-2) + ks2)) % ((-2) + ks2))) + ks2*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ((-4)*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2)))) + ((-2)*x4*ks2*ks2) + 4*ks1*x4 + 8*ks2*x4 + ks1*x4*ks2*ks2 + ((-4)*ks1*ks2*x4) + ((x0 % ((-2) + ks2)))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (((-8)*x4) + ((-2)*(((x0 // ((-2) + ks2)) % ((-2) + ks2)))) + 4*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ks2*(((x0 // ((-2) + ks2)) % ((-2) + ks2))) + ks2*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2))) + ((-4)*ks2*(triton_helpers.div_floor_integer(x0,  4 + ks2*ks2 + ((-4)*ks2)))) + ((-2)*x4*ks2*ks2) + 4*ks1*x4 + 8*ks2*x4 + ks1*x4*ks2*ks2 + ((-4)*ks1*ks2*x4) + ((x0 % ((-2) + ks2)))), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x7), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr0 + (x8), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = tmp4 >= tmp1
    tmp6 = 1.0
    tmp7 = tmp4 <= tmp6
    tmp8 = tmp5 & tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.25
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp15 = tl.where(tmp8, tmp14, tmp1)
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tl.where(tmp3, tmp17, tmp15)
    tmp19 = tl.where(tmp2, tmp1, tmp18)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 * tmp22
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (x8), tmp29, xmask)
