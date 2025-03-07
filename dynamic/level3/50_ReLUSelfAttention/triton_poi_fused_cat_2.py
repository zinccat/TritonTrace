# From: 50_ReLUSelfAttention

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 2304
    x1 = (xindex // 2304) % 1024
    x2 = (xindex // 2359296)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((ks0*x1) + (1024*ks0*((x0 // ks0) % ks1)) + (1024*ks1*ks0*x2) + (x0 % ks0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x1 + (1024*(((-768) + x0) % (ks1*ks0))) + (1024*ks1*ks0*x2)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 2304, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((ks0*x1) + (1024*ks0*((((-1536) + x0) // ks0) % ks1)) + (1024*ks1*ks0*x2) + (((-1536) + x0) % ks0)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
