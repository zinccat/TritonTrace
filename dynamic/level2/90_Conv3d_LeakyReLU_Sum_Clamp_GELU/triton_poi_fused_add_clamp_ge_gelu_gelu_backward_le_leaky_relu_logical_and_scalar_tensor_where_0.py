# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_ge_gelu_gelu_backward_le_leaky_relu_logical_and_scalar_tensor_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_ge_gelu_gelu_backward_le_leaky_relu_logical_and_scalar_tensor_where_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // ks0) % 16)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tmp8 = -1.0
    tmp9 = tmp7 >= tmp8
    tmp10 = 1.0
    tmp11 = tmp7 <= tmp10
    tmp12 = tmp9 & tmp11
    tmp14 = triton_helpers.maximum(tmp7, tmp8)
    tmp15 = triton_helpers.minimum(tmp14, tmp10)
    tmp16 = 0.7071067811865476
    tmp17 = tmp15 * tmp16
    tmp18 = libdevice.erf(tmp17)
    tmp19 = tmp18 + tmp10
    tmp20 = 0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 * tmp15
    tmp23 = -0.5
    tmp24 = tmp22 * tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = 0.3989422804014327
    tmp27 = tmp25 * tmp26
    tmp28 = tmp15 * tmp27
    tmp29 = tmp21 + tmp28
    tmp30 = tmp13 * tmp29
    tmp31 = tl.where(tmp12, tmp30, tmp1)
    tl.store(out_ptr0 + (x3), tmp31, xmask)
