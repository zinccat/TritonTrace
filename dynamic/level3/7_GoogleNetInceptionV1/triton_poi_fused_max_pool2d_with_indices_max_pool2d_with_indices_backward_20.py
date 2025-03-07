# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i8', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=82), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0) % ks0
    x2 = (xindex // ks1)
    x5 = xindex % ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))))) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))), xmask, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr1 + (x2 + (x2*((triton_helpers.div_floor_integer((-1) + ks2,  16))*(triton_helpers.div_floor_integer((-1) + ks2,  16)))) + ((triton_helpers.div_floor_integer((-1) + ks2,  16))*((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))) + (2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  16))) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))))) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3
    tmp5 = (-1) + ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))
    tmp6 = tmp5 + tmp2
    tmp7 = (-1) + ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))
    tmp8 = tmp7 + tmp4
    tmp9 = ks0
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8
    tmp13 = x5
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tmp18 = tl.where((tmp17 < 0) != (tmp1 < 0), tl.where(tmp17 % tmp1 != 0, tmp17 // tmp1 - 1, tmp17 // tmp1), tmp17 // tmp1)
    tmp19 = tmp18 * tmp1
    tmp20 = tmp17 - tmp19
    tmp21 = tmp5 + tmp18
    tmp22 = (-1) + ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))
    tmp23 = tmp22 + tmp20
    tmp24 = tmp21 * tmp9
    tmp25 = tmp24 + tmp23
    tmp27 = tmp25 == tmp13
    tmp28 = ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp29 = ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))
    tmp30 = tmp28 < tmp29
    tmp31 = 1 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))
    tmp32 = ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))
    tmp33 = tmp31 < tmp32
    tmp34 = tmp30 & tmp33
    tmp35 = tmp34 & tmp27
    tmp36 = tmp16 + tmp26
    tmp37 = tl.where(tmp35, tmp36, tmp16)
    tmp39 = tl.where((tmp38 < 0) != (tmp1 < 0), tl.where(tmp38 % tmp1 != 0, tmp38 // tmp1 - 1, tmp38 // tmp1), tmp38 // tmp1)
    tmp40 = tmp39 * tmp1
    tmp41 = tmp38 - tmp40
    tmp42 = tmp5 + tmp39
    tmp43 = (-1) + ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0))))))
    tmp44 = tmp43 + tmp41
    tmp45 = tmp42 * tmp9
    tmp46 = tmp45 + tmp44
    tmp48 = tmp46 == tmp13
    tmp49 = 2 + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))
    tmp50 = tmp49 < tmp32
    tmp51 = tmp30 & tmp50
    tmp52 = tmp51 & tmp48
    tmp53 = tmp37 + tmp47
    tmp54 = tl.where(tmp52, tmp53, tmp37)
    tmp56 = tl.where((tmp55 < 0) != (tmp1 < 0), tl.where(tmp55 % tmp1 != 0, tmp55 // tmp1 - 1, tmp55 // tmp1), tmp55 // tmp1)
    tmp57 = tmp56 * tmp1
    tmp58 = tmp55 - tmp57
    tmp59 = (-1) + ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))
    tmp60 = tmp59 + tmp56
    tmp61 = tmp7 + tmp58
    tmp62 = tmp60 * tmp9
    tmp63 = tmp62 + tmp61
    tmp65 = tmp63 == tmp13
    tmp66 = 1 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp67 = tmp66 < tmp29
    tmp68 = ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))
    tmp69 = tmp68 < tmp32
    tmp70 = tmp67 & tmp69
    tmp71 = tmp70 & tmp65
    tmp72 = tmp54 + tmp64
    tmp73 = tl.where(tmp71, tmp72, tmp54)
    tmp75 = tl.where((tmp74 < 0) != (tmp1 < 0), tl.where(tmp74 % tmp1 != 0, tmp74 // tmp1 - 1, tmp74 // tmp1), tmp74 // tmp1)
    tmp76 = tmp75 * tmp1
    tmp77 = tmp74 - tmp76
    tmp78 = tmp59 + tmp75
    tmp79 = tmp22 + tmp77
    tmp80 = tmp78 * tmp9
    tmp81 = tmp80 + tmp79
    tmp83 = tmp81 == tmp13
    tmp84 = tmp67 & tmp33
    tmp85 = tmp84 & tmp83
    tmp86 = tmp73 + tmp82
    tmp87 = tl.where(tmp85, tmp86, tmp73)
    tmp89 = tl.where((tmp88 < 0) != (tmp1 < 0), tl.where(tmp88 % tmp1 != 0, tmp88 // tmp1 - 1, tmp88 // tmp1), tmp88 // tmp1)
    tmp90 = tmp89 * tmp1
    tmp91 = tmp88 - tmp90
    tmp92 = tmp59 + tmp89
    tmp93 = tmp43 + tmp91
    tmp94 = tmp92 * tmp9
    tmp95 = tmp94 + tmp93
    tmp97 = tmp95 == tmp13
    tmp98 = tmp67 & tmp50
    tmp99 = tmp98 & tmp97
    tmp100 = tmp87 + tmp96
    tmp101 = tl.where(tmp99, tmp100, tmp87)
    tmp103 = tl.where((tmp102 < 0) != (tmp1 < 0), tl.where(tmp102 % tmp1 != 0, tmp102 // tmp1 - 1, tmp102 // tmp1), tmp102 // tmp1)
    tmp104 = tmp103 * tmp1
    tmp105 = tmp102 - tmp104
    tmp106 = (-1) + ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) * ((2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) <= ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0))))) + ((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) * (((-1) + ((ks0) * ((ks0) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (ks0)))) < (2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))))))
    tmp107 = tmp106 + tmp103
    tmp108 = tmp7 + tmp105
    tmp109 = tmp107 * tmp9
    tmp110 = tmp109 + tmp108
    tmp112 = tmp110 == tmp13
    tmp113 = 2 + ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp114 = tmp113 < tmp29
    tmp115 = tmp114 & tmp69
    tmp116 = tmp115 & tmp112
    tmp117 = tmp101 + tmp111
    tmp118 = tl.where(tmp116, tmp117, tmp101)
    tmp120 = tl.where((tmp119 < 0) != (tmp1 < 0), tl.where(tmp119 % tmp1 != 0, tmp119 // tmp1 - 1, tmp119 // tmp1), tmp119 // tmp1)
    tmp121 = tmp120 * tmp1
    tmp122 = tmp119 - tmp121
    tmp123 = tmp106 + tmp120
    tmp124 = tmp22 + tmp122
    tmp125 = tmp123 * tmp9
    tmp126 = tmp125 + tmp124
    tmp128 = tmp126 == tmp13
    tmp129 = tmp114 & tmp33
    tmp130 = tmp129 & tmp128
    tmp131 = tmp118 + tmp127
    tmp132 = tl.where(tmp130, tmp131, tmp118)
    tmp134 = tl.where((tmp133 < 0) != (tmp1 < 0), tl.where(tmp133 % tmp1 != 0, tmp133 // tmp1 - 1, tmp133 // tmp1), tmp133 // tmp1)
    tmp135 = tmp134 * tmp1
    tmp136 = tmp133 - tmp135
    tmp137 = tmp106 + tmp134
    tmp138 = tmp43 + tmp136
    tmp139 = tmp137 * tmp9
    tmp140 = tmp139 + tmp138
    tmp142 = tmp140 == tmp13
    tmp143 = tmp114 & tmp50
    tmp144 = tmp143 & tmp142
    tmp145 = tmp132 + tmp141
    tmp146 = tl.where(tmp144, tmp145, tmp132)
    tl.store(out_ptr0 + (x4), tmp146, xmask)
