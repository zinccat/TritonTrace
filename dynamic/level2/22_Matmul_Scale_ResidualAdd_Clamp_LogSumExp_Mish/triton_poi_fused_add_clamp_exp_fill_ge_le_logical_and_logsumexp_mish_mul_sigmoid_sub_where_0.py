# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_exp_fill_ge_le_logical_and_logsumexp_mish_mul_sigmoid_sub_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_exp_fill_ge_le_logical_and_logsumexp_mish_mul_sigmoid_sub_where_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = 20.0
    tmp3 = tmp1 > tmp2
    tmp4 = tl_math.exp(tmp1)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp0 * tmp8
    tmp10 = tmp0 * tmp1
    tmp11 = tl.sigmoid(tmp1)
    tmp12 = tmp1 * tmp11
    tmp13 = tmp7 * tmp7
    tmp14 = 1.0
    tmp15 = tmp14 - tmp13
    tmp16 = tmp12 * tmp15
    tmp17 = tmp7 + tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp9 + tmp18
    tmp21 = 2.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 + tmp22
    tmp24 = -10.0
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = 10.0
    tmp27 = triton_helpers.minimum(tmp25, tmp26)
    tmp28 = tmp27 - tmp1
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp19 * tmp29
    tmp31 = tmp23 >= tmp24
    tmp32 = tmp23 <= tmp26
    tmp33 = tmp31 & tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp33, tmp30, tmp34)
    tmp36 = tmp35 + tmp35
    tmp37 = tmp36 * tmp21
    tl.store(in_out_ptr0 + (x2), tmp37, xmask)
