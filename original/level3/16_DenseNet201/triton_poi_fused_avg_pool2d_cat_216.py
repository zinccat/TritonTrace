# From: 16_DenseNet201

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_cat_216', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_cat_216(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, xnumel, XBLOCK : tl.constexpr):
    xnumel = 439040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = xindex // 7
    x4 = xindex
    x2 = (xindex % 43904)
    x3 = xindex // 43904
    tmp0 = tl.load(in_ptr0 + (2*x0 + 28*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 28*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + 2*x0 + 28*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + 2*x0 + 28*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x2 + 56448*x3), tmp8, xmask)
    tl.store(out_ptr2 + (x2 + 58016*x3), tmp8, xmask)
    tl.store(out_ptr3 + (x2 + 59584*x3), tmp8, xmask)
    tl.store(out_ptr4 + (x2 + 61152*x3), tmp8, xmask)
    tl.store(out_ptr5 + (x2 + 62720*x3), tmp8, xmask)
    tl.store(out_ptr6 + (x2 + 64288*x3), tmp8, xmask)
    tl.store(out_ptr7 + (x2 + 65856*x3), tmp8, xmask)
    tl.store(out_ptr8 + (x2 + 67424*x3), tmp8, xmask)
    tl.store(out_ptr9 + (x2 + 68992*x3), tmp8, xmask)
    tl.store(out_ptr10 + (x2 + 70560*x3), tmp8, xmask)
    tl.store(out_ptr11 + (x2 + 72128*x3), tmp8, xmask)
    tl.store(out_ptr12 + (x2 + 73696*x3), tmp8, xmask)
    tl.store(out_ptr13 + (x2 + 75264*x3), tmp8, xmask)
    tl.store(out_ptr14 + (x2 + 76832*x3), tmp8, xmask)
    tl.store(out_ptr15 + (x2 + 78400*x3), tmp8, xmask)
    tl.store(out_ptr16 + (x2 + 79968*x3), tmp8, xmask)
    tl.store(out_ptr17 + (x2 + 81536*x3), tmp8, xmask)
    tl.store(out_ptr18 + (x2 + 83104*x3), tmp8, xmask)
    tl.store(out_ptr19 + (x2 + 84672*x3), tmp8, xmask)
    tl.store(out_ptr20 + (x2 + 86240*x3), tmp8, xmask)
    tl.store(out_ptr21 + (x2 + 87808*x3), tmp8, xmask)
    tl.store(out_ptr22 + (x2 + 89376*x3), tmp8, xmask)
    tl.store(out_ptr23 + (x2 + 90944*x3), tmp8, xmask)
    tl.store(out_ptr24 + (x2 + 92512*x3), tmp8, xmask)
    tl.store(out_ptr25 + (x2 + 94080*x3), tmp8, xmask)
