# From: 48_Mamba2ReturnY

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_masked_fill_mul_sub_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x7 = xindex
    x9 = xindex % ks0
    x1 = (xindex // ks1) % ks1
    x2 = (xindex // ks0) % 8
    x3 = (xindex // ks2) % ks3
    x4 = (xindex // ks4)
    x0 = xindex % ks1
    x11 = (xindex // ks2)
    x5 = (xindex // ks0) % ks3
    x6 = (xindex // ks5) % 8
    x12 = (xindex // ks1)
    x13 = (xindex // ks6) % ks7
    tmp0 = tl.load(in_ptr0 + (x7), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x9), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x1 + (ks1*x3) + (ks1*ks3*x2) + (8*ks1*ks3*x4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (ks1*x3) + (ks1*ks3*x2) + (8*ks1*ks3*x4)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x9 + (ks0*x6) + (8*ks0*x5) + (8*ks0*ks3*x4)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x9 + (ks0*x6) + (8*ks0*x5) + (8*ks0*ks3*x4)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x12), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0 + (ks1*x13)), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp1, tmp5, tmp4)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp0 * tmp7
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 - tmp13
    tmp15 = tl.where(tmp1, tmp5, tmp14)
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp11 * tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp1, tmp18, tmp17)
    tl.store(out_ptr0 + (x0 + (ks1*x2) + (8*ks0*x11) + (8*ks1*x1)), tmp8, xmask)
    tl.store(out_ptr1 + (x7), tmp19, xmask)
