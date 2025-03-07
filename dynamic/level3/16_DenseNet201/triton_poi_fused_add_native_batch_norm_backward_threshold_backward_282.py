# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32', 23: 'i32', 24: 'i32', 25: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_282', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, ks0, ks1, ks2, ks3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex % ks0
    x5 = (xindex // ks0)
    x6 = xindex
    x3 = (xindex // ks2) % 128
    tmp0 = tl.load(in_ptr0 + (x4 + (512*x5) + (512*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (1024*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + (480*x5) + (480*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (960*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 + (448*x5) + (448*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (896*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 + (416*x5) + (416*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (832*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x4 + (384*x5) + (384*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (768*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x4 + (352*x5) + (352*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (704*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x4 + (320*x5) + (320*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (640*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x4 + (288*x5) + (288*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (576*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (x4 + (256*x5) + (256*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (512*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x4 + (224*x5) + (224*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (448*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr10 + (x4 + (192*x5) + (192*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (384*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x4 + (160*x5) + (160*x5*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (320*x5*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr12 + (x6), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr13 + (x6), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr14 + (x6), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr15 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr17 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr18 + (x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr19 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = 0.0
    tmp25 = tmp23 <= tmp24
    tmp27 = tl.where(tmp25, tmp24, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = 1.00000000000000 / (((128*ks3) + (128*ks3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) + (256*ks3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)))) / 128)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 * tmp36
    tmp38 = tmp30 * tmp37
    tmp39 = tmp27 - tmp38
    tmp41 = tmp40 * tmp33
    tmp42 = tmp39 - tmp41
    tmp44 = tmp35 * tmp43
    tmp45 = tmp42 * tmp44
    tmp46 = tmp22 + tmp45
    tl.store(in_out_ptr0 + (x6), tmp46, xmask)
