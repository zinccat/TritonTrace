# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: 'i32', 28: 'i32', 29: 'i32', 30: 'i32', 31: 'i32', 32: 'i32', 33: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32, 33), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_cat_214', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks0
    x2 = (xindex // ks1)
    x4 = (xindex // ks5)
    x5 = xindex % ks5
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (ks2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + ks2 + (2*x0) + (ks3*x2) + (2*ks2*x1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x0 + x1 + x2 + (x1*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks4,  4)),  2)),  2)),  2))) + (x2*((triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks4,  4)),  2)),  2)),  2))*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks4,  4)),  2)),  2)),  2)))) + (2*x2*(triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + (triton_helpers.div_floor_integer((-1) + ks4,  4)),  2)),  2)),  2)))), tmp8, xmask)
    tl.store(out_ptr1 + (x5 + (1152*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr2 + (x5 + (1184*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr3 + (x5 + (1216*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr4 + (x5 + (1248*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr5 + (x5 + (1280*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr6 + (x5 + (1312*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr7 + (x5 + (1344*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr8 + (x5 + (1376*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr9 + (x5 + (1408*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr10 + (x5 + (1440*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr11 + (x5 + (1472*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr12 + (x5 + (1504*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr13 + (x5 + (1536*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr14 + (x5 + (1568*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr15 + (x5 + (1600*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr16 + (x5 + (1632*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr17 + (x5 + (1664*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr18 + (x5 + (1696*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr19 + (x5 + (1728*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr20 + (x5 + (1760*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr21 + (x5 + (1792*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr22 + (x5 + (1824*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr23 + (x5 + (1856*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr24 + (x5 + (1888*ks1*x4)), tmp8, xmask)
    tl.store(out_ptr25 + (x5 + (1920*ks1*x4)), tmp8, xmask)
