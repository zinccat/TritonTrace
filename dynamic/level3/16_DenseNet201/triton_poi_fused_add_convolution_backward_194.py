# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: 'i32', 34: 'i32', 35: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_194', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % ks0
    x3 = (xindex // ks0)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1792*x3) + (1792*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3584*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1760*x3) + (1760*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3520*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1728*x3) + (1728*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3456*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1696*x3) + (1696*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3392*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1664*x3) + (1664*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3328*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1632*x3) + (1632*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3264*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1600*x3) + (1600*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3200*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1568*x3) + (1568*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3136*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1536*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1536*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3072*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1504*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1504*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (3008*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr10 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1472*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1472*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2944*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1440*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1440*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2880*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr12 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1408*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1408*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2816*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1376*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1376*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2752*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1344*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1344*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2688*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr15 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1312*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1312*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2624*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1280*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1280*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2560*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr17 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1248*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1248*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2496*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr18 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1216*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1216*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2432*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1184*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1184*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2368*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr20 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1152*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1152*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2304*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1120*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1120*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2240*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr22 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1088*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1088*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2176*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr23 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1056*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1056*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2112*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1024*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (1024*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (2048*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr25 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (992*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (992*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1984*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr26 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (960*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (960*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1920*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr27 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (928*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (928*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1856*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr28 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (896*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (896*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1792*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr29 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (864*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (864*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1728*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr30 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (832*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (832*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1664*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (768 + x2 + (768*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (800*x3) + (1536*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))) + (800*x3*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))) + (1600*x3*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks1,  4)),  2)),  2)))), xmask, eviction_policy='evict_last')
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
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp44 = tmp42 + tmp43
    tmp46 = tmp44 + tmp45
    tmp48 = tmp46 + tmp47
    tmp50 = tmp48 + tmp49
    tmp52 = tmp50 + tmp51
    tmp54 = tmp52 + tmp53
    tmp56 = tmp54 + tmp55
    tmp58 = tmp56 + tmp57
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tl.store(in_out_ptr0 + (x4), tmp62, xmask)
