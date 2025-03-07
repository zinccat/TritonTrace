# From: 45_UNetSoftmax

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i8', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks1
    x2 = (xindex // ks2)
    x5 = xindex % ks2
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((ks3*((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) * ((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) <= ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4))))) + ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) * (((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) < (((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0))))))) + (ks3*ks4*x2) + ((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) * ((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) <= ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3))))) + ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) * (((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) < (((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0))))))), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + ((ks3*((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) * ((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) <= ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4))))) + ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) * (((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) < (((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0))))))) + (ks3*ks4*x2) + ((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) * ((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) <= ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3))))) + ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) * (((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) < (((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0))))))), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = 2*((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) * ((((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0)))) <= ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4))))) + ((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) * (((-1) + ((ks4) * ((ks4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (ks4)))) < (((0) * ((0) >= ((x1 // 2))) + ((x1 // 2)) * (((x1 // 2)) > (0))))))
    tmp6 = tmp5 + tmp2
    tmp7 = 2*((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) * ((((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0)))) <= ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3))))) + ((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) * (((-1) + ((ks3) * ((ks3) <= (1 + (x0 // 2))) + (1 + (x0 // 2)) * ((1 + (x0 // 2)) < (ks3)))) < (((0) * ((0) >= ((x0 // 2))) + ((x0 // 2)) * (((x0 // 2)) > (0))))))
    tmp8 = tmp7 + tmp4
    tmp9 = ks0
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp13 = x5
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
