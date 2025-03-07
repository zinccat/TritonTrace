# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_24poi_fused_max_pool2d_with_indices_24(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 86528
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    x0 = (index % 512)
    x1 = ((index // 512) % 13)
    x2 = index // 6656
    x3 = index

    input_val0 = tl.load(input_ptr + (x0 + 1024*x1 + 27648*x2), mask)
    input_val1 = tl.load(input_ptr + (512 + x0 + 1024*x1 + 27648*x2), mask)
    input_val3 = tl.load(input_ptr + (1024 + x0 + 1024*x1 + 27648*x2), mask)
    input_val5 = tl.load(input_ptr + (13824 + x0 + 1024*x1 + 27648*x2), mask)
    input_val7 = tl.load(input_ptr + (14336 + x0 + 1024*x1 + 27648*x2), mask)
    input_val9 = tl.load(input_ptr + (14848 + x0 + 1024*x1 + 27648*x2), mask)
    input_val11 = tl.load(input_ptr + (27648 + x0 + 1024*x1 + 27648*x2), mask)
    input_val13 = tl.load(input_ptr + (28160 + x0 + 1024*x1 + 27648*x2), mask)
    input_val15 = tl.load(input_ptr + (28672 + x0 + 1024*x1 + 27648*x2), mask)

    max_val2 = triton_helpers.maximum(input_val1, input_val0)
    max_val4 = triton_helpers.maximum(input_val3, max_val2)
    max_val6 = triton_helpers.maximum(input_val5, max_val4)
    max_val8 = triton_helpers.maximum(input_val7, max_val6)
    max_val10 = triton_helpers.maximum(input_val9, max_val8)
    max_val12 = triton_helpers.maximum(input_val11, max_val10)
    max_val14 = triton_helpers.maximum(input_val13, max_val12)
    max_val16 = triton_helpers.maximum(input_val15, max_val14)

    index_val1 = input_val1 > input_val0
    index_val18 = tl.full([1], 1, tl.int8)
    index_val19 = tl.full([1], 0, tl.int8)
    index_val20 = tl.where(index_val1, index_val18, index_val19)

    index_val21 = input_val3 > max_val2
    index_val22 = tl.full([1], 2, tl.int8)
    index_val23 = tl.where(index_val21, index_val22, index_val20)

    index_val24 = input_val5 > max_val4
    index_val25 = tl.full([1], 3, tl.int8)
    index_val26 = tl.where(index_val24, index_val25, index_val23)

    index_val27 = input_val7 > max_val6
    index_val28 = tl.full([1], 4, tl.int8)
    index_val29 = tl.where(index_val27, index_val28, index_val26)

    index_val30 = input_val9 > max_val8
    index_val31 = tl.full([1], 5, tl.int8)
    index_val32 = tl.where(index_val30, index_val31, index_val29)

    index_val33 = input_val11 > max_val10
    index_val34 = tl.full([1], 6, tl.int8)
    index_val35 = tl.where(index_val33, index_val34, index_val32)

    index_val36 = input_val13 > max_val12
    index_val37 = tl.full([1], 7, tl.int8)
    index_val38 = tl.where(index_val36, index_val37, index_val35)

    index_val39 = input_val15 > max_val14
    index_val40 = tl.full([1], 8, tl.int8)
    index_val41 = tl.where(index_val39, index_val40, index_val38)

    tl.store(output_ptr_max + (x3), max_val16, mask)
    tl.store(output_ptr_indices + (x3), index_val41, mask)