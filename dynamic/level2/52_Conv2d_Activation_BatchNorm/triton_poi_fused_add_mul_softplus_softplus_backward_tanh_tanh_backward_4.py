# From: 52_Conv2d_Activation_BatchNorm

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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_softplus_softplus_backward_tanh_tanh_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_softplus_softplus_backward_tanh_tanh_backward_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = 20.0
    tmp3 = tmp1 > tmp2
    tmp4 = tl_math.exp(tmp1)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp0 * tmp7
    tmp9 = 1.0
    tmp10 = tmp1 * tmp9
    tmp11 = tmp10 > tmp2
    tmp12 = tmp0 * tmp1
    tmp13 = tmp7 * tmp7
    tmp14 = tmp9 - tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tl_math.exp(tmp10)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp16 + tmp9
    tmp19 = tmp17 / tmp18
    tmp20 = tl.where(tmp11, tmp15, tmp19)
    tmp21 = tmp8 + tmp20
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
