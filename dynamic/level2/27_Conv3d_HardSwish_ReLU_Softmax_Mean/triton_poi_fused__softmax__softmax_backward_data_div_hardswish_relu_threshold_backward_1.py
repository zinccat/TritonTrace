# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__softmax_backward_data_div_hardswish_relu_threshold_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_div_hardswish_relu_threshold_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex % ks0)
    x6 = xindex // ks1
    x7 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (x4 + ((-8)*x6) + ((-2)*x6*ks3*ks3) + 4*ks2*x6 + 8*ks3*x6 + ks2*x6*ks3*ks3 + ((-4)*ks2*ks3*x6)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x4 + ((-8)*x6) + ((-2)*x6*ks3*ks3) + 4*ks2*x6 + 8*ks3*x6 + ks2*x6*ks3*ks3 + ((-4)*ks2*ks3*x6)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x4 + ((-8)*x6) + ((-2)*x6*ks3*ks3) + 4*ks2*x6 + 8*ks3*x6 + ks2*x6*ks3*ks3 + ((-4)*ks2*ks3*x6)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x7), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tmp11 <= tmp3
    tmp14 = tmp11 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp17 = tmp15 / tmp16
    tmp18 = -tmp17
    tmp21 = ks0
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp23 * tmp17
    tmp25 = libdevice.fma(tmp18, tmp19, tmp24)
    tmp26 = tl.where(tmp12, tmp3, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, xmask)
