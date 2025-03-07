# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

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
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_scalar_tensor_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_scalar_tensor_where_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks0)
    x2 = xindex // ks1
    x5 = (xindex % ks1)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + ((-1)*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0))))))) + x2*(ks2 // 2)*(ks2 // 2) + (ks2 // 2)*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + ((-2)*x2*(ks2 // 2)) + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x2 + ((-1)*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0))))))) + x2*(ks2 // 2)*(ks2 // 2) + (ks2 // 2)*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + ((-2)*x2*(ks2 // 2)) + ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0))))))
    tmp6 = tmp5 + tmp2
    tmp7 = 2*((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) * ((((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0)))) <= ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2)))))) + ((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) * (((-1) + (((-1) + (ks2 // 2)) * (((-1) + (ks2 // 2)) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < ((-1) + (ks2 // 2))))) < (((0) * ((0) >= (x0 // 2)) + (x0 // 2) * ((x0 // 2) > (0))))))
    tmp8 = tmp7 + tmp4
    tmp9 = ks0
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp13 = x5
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
