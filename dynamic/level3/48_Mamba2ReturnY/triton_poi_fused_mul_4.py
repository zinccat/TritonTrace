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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x7 = xindex % ks0
    x1 = (xindex // ks1) % ks1
    x2 = (xindex // ks0) % 8
    x3 = (xindex // ks3) % ks2
    x4 = (xindex // ks4)
    x0 = xindex % ks1
    x9 = (xindex // ks3)
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x1 + (ks1*x3) + (ks1*ks2*x2) + (8*ks1*ks2*x4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (ks1*x3) + (ks1*ks2*x2) + (8*ks1*ks2*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp1, tmp5, tmp4)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp0 * tmp7
    tl.store(out_ptr0 + (x0 + (ks1*x2) + (8*ks0*x9) + (8*ks1*x1)), tmp8, xmask)
