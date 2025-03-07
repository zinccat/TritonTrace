# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_exp_logsumexp_max_mean_mul_scatter_sub_zeros_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_exp_logsumexp_max_mean_mul_scatter_sub_zeros_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp13 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 1.0
    tmp3 = tmp1 / tmp2
    tmp4 = tl_math.abs(tmp3)
    tmp5 = float("inf")
    tmp6 = tmp4 == tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp3)
    tmp9 = tmp3 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp11 + tmp8
    tmp14 = tmp12 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp0 * tmp15
    tmp17 = tmp3 - tmp12
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19 * tmp2
    tl.store(out_ptr0 + (x0), tmp20, xmask)
