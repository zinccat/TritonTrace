# From: 28_VisionTransformer

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 3
    x0 = xindex % 512
    x2 = (xindex // 1536) % ks0
    x3 = (xindex // ks1)
    x4 = xindex % 1536
    tmp3 = tl.load(in_ptr0 + ((64*((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) % 8)) + (512*(((x0 + (512*x2) + (512*x3) + (512*x2*((ks3 // 16)*(ks3 // 16)))) // (512 + (512*((ks3 // 16)*(ks3 // 16))))) % ks0)) + (512*ks0*(((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) // 8) % ks2)) + (x0 % 64)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + ((64*((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) % 8)) + (512*(((x0 + (512*x2) + (512*x3) + (512*x2*((ks3 // 16)*(ks3 // 16)))) // (512 + (512*((ks3 // 16)*(ks3 // 16))))) % ks0)) + (512*ks0*(((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) // 8) % ks2)) + (x0 % 64)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + ((64*((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) % 8)) + (512*(((x0 + (512*x2) + (512*x3) + (512*x2*((ks3 // 16)*(ks3 // 16)))) // (512 + (512*((ks3 // 16)*(ks3 // 16))))) % ks0)) + (512*ks0*(((((x0 + (512*x3)) // 64) % (8 + (8*((ks3 // 16)*(ks3 // 16))))) // 8) % ks2)) + (x0 % 64)), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x4 + (1536*x2) + (1536*x3) + (1536*x2*((ks3 // 16)*(ks3 // 16)))), tmp15, xmask)
