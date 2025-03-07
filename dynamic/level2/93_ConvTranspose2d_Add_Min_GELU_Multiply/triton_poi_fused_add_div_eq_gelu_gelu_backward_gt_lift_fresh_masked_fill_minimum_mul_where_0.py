# From: 93_ConvTranspose2d_Add_Min_GELU_Multiply

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_eq_gelu_gelu_backward_gt_lift_fresh_masked_fill_minimum_mul_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_eq_gelu_gelu_backward_gt_lift_fresh_masked_fill_minimum_mul_where_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = tmp2 == tmp3
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp9 = triton_helpers.minimum(tmp2, tmp3)
    tmp10 = 0.7071067811865476
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 * tmp1
    tmp16 = tmp9 * tmp9
    tmp17 = -0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp20 = 0.3989422804014327
    tmp21 = tmp19 * tmp20
    tmp22 = tmp9 * tmp21
    tmp23 = tmp15 + tmp22
    tmp24 = tmp8 * tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tl.where(tmp5, tmp25, tmp24)
    tmp27 = tl.where(tmp4, tmp3, tmp26)
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
