# From: 29_SwinMLP

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_51', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 294
    x2 = (xindex // 9408)
    x0 = xindex % 32
    x3 = xindex
    tmp0 = (7*((x2 // (triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))) % 5)) + ((x1 % 49) // 7)
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (7*(x2 % (triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32))))) + ((x1 % 49) % 7)
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.broadcast_to((-3) + (7*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))), [XBLOCK])
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + (4608 + x0 + ((-37632)*((x2 // (5*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32))))) % (libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))) + ((-9408)*((x2 // (triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))) % 5)) + ((-5376)*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))) + ((-1344)*((x1 % 49) // 7)) + (32*(x1 // 49)) + (192*((x1 % 49) % 7)) + (1344*(x2 % (triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32))))) + (1344*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))*((x1 % 49) // 7)) + (9408*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))*((x2 // (triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))) % 5)) + (37632*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32)))*((x2 // (5*(triton_helpers.div_floor_integer(5*ks0,  libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32))))) % (libdevice.trunc(((25*ks0).to(tl.float64)) / 25.0000000000000).to(tl.int32))))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 0.0
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp16, tmp13)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
