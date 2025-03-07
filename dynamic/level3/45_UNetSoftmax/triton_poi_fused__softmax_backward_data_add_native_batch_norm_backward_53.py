# From: 45_UNetSoftmax

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_add_native_batch_norm_backward_53', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x1 = (xindex // ks0) % ks1
    x6 = (xindex // ks2)
    x0 = xindex % ks0
    x2 = (xindex // ks2) % 128
    x3 = (xindex // ks4)
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1 + (8*ks3*x6)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (8*ks5*x1) + (8192*ks3*ks5) + (64*ks3*ks5*x2) + (16384*ks3*ks5*x3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 * tmp0
    tmp7 = libdevice.fma(tmp1, tmp2, tmp6)
    tmp10 = tmp8 - tmp9
    tmp12 = 1.00000000000000 / ((1024*ks0*ks1) / 128)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(in_out_ptr0 + (x5), tmp25, xmask)
