# From: 30_SwinTransformerV2

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_111', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 96)
    x0 = xindex % 96
    x2 = xindex
    tl.device_assert(((7*((x1 // 49) % (triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((3 + (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((x1 % 49) % 7)) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))) < 7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))), "index out of bounds: ((7*((x1 // 49) % (triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((3 + (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((x1 % 49) % 7)) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))) < 7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))")
    tmp1 = tl.load(in_ptr0 + (x0 + (96*(((7*((x1 // 49) % (triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((3 + (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) + ((x1 % 49) % 7)) % (7*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))))) + (301056*((x1 // (392*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % (libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))) + (672*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32)))*(tl.where(((3 + (7*((x1 // (49*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % 8)) + ((x1 % 49) // 7)) % 56) < 0, 56 + ((3 + (7*((x1 // (49*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % 8)) + ((x1 % 49) // 7)) % 56), (3 + (7*((x1 // (49*(triton_helpers.div_floor_integer(8*ks0,  libdevice.trunc(((64*ks0).to(tl.float64)) / 64.0000000000000).to(tl.int32))))) % 8)) + ((x1 % 49) // 7)) % 56)))), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp1, None)
