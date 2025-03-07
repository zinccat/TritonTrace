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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'out_ptr41': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_cat_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '5A06A9183D03767BDAB0FC92F89F8279B36CCC7C4B95A264F6D3CCE126D2D3A0', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_cat_69(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 14)
    x1 = xindex // 14
    x4 = xindex
    x2 = (xindex % 50176)
    x3 = xindex // 50176
    tmp0 = tl.load(in_ptr0 + (2*x0 + 56*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 56*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x0 + 56*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x0 + 56*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x2 + 100352*x3), tmp8, xmask)
    tl.store(out_ptr2 + (x2 + 106624*x3), tmp8, xmask)
    tl.store(out_ptr3 + (x2 + 112896*x3), tmp8, xmask)
    tl.store(out_ptr4 + (x2 + 119168*x3), tmp8, xmask)
    tl.store(out_ptr5 + (x2 + 125440*x3), tmp8, xmask)
    tl.store(out_ptr6 + (x2 + 131712*x3), tmp8, xmask)
    tl.store(out_ptr7 + (x2 + 137984*x3), tmp8, xmask)
    tl.store(out_ptr8 + (x2 + 144256*x3), tmp8, xmask)
    tl.store(out_ptr9 + (x2 + 150528*x3), tmp8, xmask)
    tl.store(out_ptr10 + (x2 + 156800*x3), tmp8, xmask)
    tl.store(out_ptr11 + (x2 + 163072*x3), tmp8, xmask)
    tl.store(out_ptr12 + (x2 + 169344*x3), tmp8, xmask)
    tl.store(out_ptr13 + (x2 + 175616*x3), tmp8, xmask)
    tl.store(out_ptr14 + (x2 + 181888*x3), tmp8, xmask)
    tl.store(out_ptr15 + (x2 + 188160*x3), tmp8, xmask)
    tl.store(out_ptr16 + (x2 + 194432*x3), tmp8, xmask)
    tl.store(out_ptr17 + (x2 + 200704*x3), tmp8, xmask)
    tl.store(out_ptr18 + (x2 + 206976*x3), tmp8, xmask)
    tl.store(out_ptr19 + (x2 + 213248*x3), tmp8, xmask)
    tl.store(out_ptr20 + (x2 + 219520*x3), tmp8, xmask)
    tl.store(out_ptr21 + (x2 + 225792*x3), tmp8, xmask)
    tl.store(out_ptr22 + (x2 + 232064*x3), tmp8, xmask)
    tl.store(out_ptr23 + (x2 + 238336*x3), tmp8, xmask)
    tl.store(out_ptr24 + (x2 + 244608*x3), tmp8, xmask)
    tl.store(out_ptr25 + (x2 + 250880*x3), tmp8, xmask)
    tl.store(out_ptr26 + (x2 + 257152*x3), tmp8, xmask)
    tl.store(out_ptr27 + (x2 + 263424*x3), tmp8, xmask)
    tl.store(out_ptr28 + (x2 + 269696*x3), tmp8, xmask)
    tl.store(out_ptr29 + (x2 + 275968*x3), tmp8, xmask)
    tl.store(out_ptr30 + (x2 + 282240*x3), tmp8, xmask)
    tl.store(out_ptr31 + (x2 + 288512*x3), tmp8, xmask)
    tl.store(out_ptr32 + (x2 + 294784*x3), tmp8, xmask)
    tl.store(out_ptr33 + (x2 + 301056*x3), tmp8, xmask)
    tl.store(out_ptr34 + (x2 + 307328*x3), tmp8, xmask)
    tl.store(out_ptr35 + (x2 + 313600*x3), tmp8, xmask)
    tl.store(out_ptr36 + (x2 + 319872*x3), tmp8, xmask)
    tl.store(out_ptr37 + (x2 + 326144*x3), tmp8, xmask)
    tl.store(out_ptr38 + (x2 + 332416*x3), tmp8, xmask)
    tl.store(out_ptr39 + (x2 + 338688*x3), tmp8, xmask)
    tl.store(out_ptr40 + (x2 + 344960*x3), tmp8, xmask)
    tl.store(out_ptr41 + (x2 + 351232*x3), tmp8, xmask)
