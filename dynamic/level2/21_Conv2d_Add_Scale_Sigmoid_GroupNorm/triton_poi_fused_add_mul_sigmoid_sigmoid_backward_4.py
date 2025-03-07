# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_sigmoid_backward_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_sigmoid_backward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x7 = xindex // ks0
    x2 = ((xindex // ks1) % 16)
    x0 = (xindex % ks2)
    x1 = ((xindex // ks2) % ks2)
    x8 = xindex // ks1
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + ((-2)*((((x0 + ((-2)*x1) + ks3*x1) // ((-2) + ks3)) % ((-2) + ks3)))) + 4*x8 + ks3*((((x0 + ((-2)*x1) + ks3*x1) // ((-2) + ks3)) % ((-2) + ks3))) + x8*ks3*ks3 + ((-4)*ks3*x8)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x7 // 2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp16 + tmp6
    tmp18 = tmp17 * tmp8
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = tmp19 * tmp21
    tmp23 = tmp15 * tmp22
    tl.store(out_ptr0 + (x4), tmp23, xmask)
