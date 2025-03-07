# From: 30_SwinTransformerV2

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_97', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = (xindex // 192)
    x5 = (xindex // ks0)
    x7 = xindex % ks0
    x0 = xindex % 192
    x8 = (xindex // 192) % ks1
    x10 = (xindex // ks2)
    x2 = (xindex // 1344) % ks3
    x3 = (xindex // ks4) % 7
    x9 = xindex % 1344
    tmp0 = tl.load(in_ptr0 + (x6), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x7 + (150528*x5)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x7 + (150528*x5)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x8 + (784*x5)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x7 + (150528*x5)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x8 + (784*x5)), xmask, eviction_policy='evict_last')
    tmp1 = 0.005208333333333333
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = 192.0
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 - tmp10
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp11 - tmp17
    tmp19 = tmp2 * tmp18
    tl.store(out_ptr0 + (x9 + (1344*x3) + (9408*x2) + (9408*x10*(triton_helpers.div_floor_integer(ks6,  libdevice.trunc((ks5.to(tl.float64)) / 16.0000000000000).to(tl.int32))))), tmp19, xmask)
