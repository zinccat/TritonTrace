# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32', 16: 'i32', 17: 'i32', 18: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_310', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % ks0
    x4 = (xindex // ks0)
    x5 = xindex
    x1 = (xindex // ks2) % 64
    tmp0 = tl.load(in_ptr0 + (x3 + (256*x4) + (256*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (512*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (224*x4) + (224*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (448*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (192*x4) + (192*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (384*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3 + (160*x4) + (160*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (320*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3 + (128*x4) + (128*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (256*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x3 + (96*x4) + (96*x4*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (192*x4*(triton_helpers.div_floor_integer((-1) + ks1,  4)))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x5), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x5), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr12 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = 0.0
    tmp13 = tmp11 <= tmp12
    tmp15 = tl.where(tmp13, tmp12, tmp14)
    tmp18 = tmp16 - tmp17
    tmp20 = 1.00000000000000 / (((64*ks3) + (64*ks3*((triton_helpers.div_floor_integer((-1) + ks1,  4))*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) + (128*ks3*(triton_helpers.div_floor_integer((-1) + ks1,  4)))) / 64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tmp15 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = tmp10 + tmp33
    tl.store(in_out_ptr0 + (x5), tmp34, xmask)
