# From: 24_Conv3d_Min_Softmax

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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_zeros_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scatter_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex % ks1)
    x6 = xindex // ks2
    x7 = xindex // ks1
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x4 + 4*x6 + x6*ks3*ks3 + ((-4)*ks3*x6)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < (-2) + ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < (-2) + ks0")
    tmp3 = -tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = libdevice.fma(tmp3, tmp4, tmp6)
    tl.store(out_ptr0 + (x4 + ((-8)*x7) + 4*tmp0 + tmp0*ks3*ks3 + ((-4)*ks3*tmp0) + ((-2)*x7*ks3*ks3) + 4*ks0*x7 + 8*ks3*x7 + ks0*x7*ks3*ks3 + ((-4)*ks0*ks3*x7)), tmp7, xmask)
