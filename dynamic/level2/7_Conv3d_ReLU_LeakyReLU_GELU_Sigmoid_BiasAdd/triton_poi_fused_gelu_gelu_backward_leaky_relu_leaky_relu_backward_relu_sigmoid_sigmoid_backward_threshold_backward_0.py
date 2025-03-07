# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_relu_sigmoid_sigmoid_backward_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_leaky_relu_leaky_relu_backward_relu_sigmoid_sigmoid_backward_threshold_backward_0(in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr1 + (x0), xmask)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 0.01
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp14 - tmp17
    tmp19 = tmp17 * tmp18
    tmp20 = tmp0 * tmp19
    tmp21 = tmp15 * tmp9
    tmp22 = tmp8 * tmp8
    tmp23 = -0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = 0.3989422804014327
    tmp27 = tmp25 * tmp26
    tmp28 = tmp8 * tmp27
    tmp29 = tmp21 + tmp28
    tmp30 = tmp20 * tmp29
    tmp31 = tmp30 * tmp6
    tmp32 = tl.where(tmp5, tmp30, tmp31)
    tmp33 = tmp3 <= tmp4
    tmp34 = tl.where(tmp33, tmp4, tmp32)
    tl.store(in_out_ptr1 + (x0), tmp34, xmask)
