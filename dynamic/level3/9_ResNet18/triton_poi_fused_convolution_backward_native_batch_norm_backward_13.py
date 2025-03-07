# From: 9_ResNet18

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32', 17: 'i32', 18: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // ks0) % 256
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.00000000000000 / (((256*ks1) + (256*ks1*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + (512*ks1*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) / 256)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 * tmp6
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp0 - tmp27
    tmp30 = tmp29 * tmp6
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
