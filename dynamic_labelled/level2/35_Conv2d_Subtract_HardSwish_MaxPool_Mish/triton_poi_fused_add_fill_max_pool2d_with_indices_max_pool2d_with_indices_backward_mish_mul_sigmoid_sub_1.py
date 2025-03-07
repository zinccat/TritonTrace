# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_max_pool2d_with_indices_max_pool2d_with_indices_backward_mish_mul_sigmoid_sub_1poi_fused_add_fill_max_pool2d_with_indices_max_pool2d_with_indices_backward_mish_mul_sigmoid_sub_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size0
    y_coord = (block_indices // kernel_size0) % kernel_size0
    z_coord = block_indices // kernel_size1
    linear_index = block_indices

    # Load input data with complex indexing
    input_index0 = (
        z_coord + ((-1) * (
            ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) *
            (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) <=
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
             (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                       (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
              (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))))
        ) * (kernel_size2 // 2) * (kernel_size2 // 2) +
        (kernel_size2 // 2) * (
            ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) *
            (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) <=
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
             (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                       (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
              (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))))
        ) + ((-2) * z_coord * (kernel_size2 // 2)) +
        (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) *
         (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) <=
          ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                   (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
          ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                   (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
          (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                    (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
           (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))))))
    )
    tmp0 = tl.load(input_ptr0 + input_index0, valid_mask, eviction_policy='evict_last')

    input_index1 = (
        z_coord + ((-1) * (
            ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) *
            (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) <=
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
             (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                       (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
              (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))))
        ) * (kernel_size2 // 2) * (kernel_size2 // 2) +
        (kernel_size2 // 2) * (
            ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) *
            (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) <=
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
             ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                      (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
             (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                       (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
              (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))))
        ) + ((-2) * z_coord * (kernel_size2 // 2)) +
        (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) *
         (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) <=
          ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                   (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
          ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                   (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
          (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                    (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
           (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))))))
    )
    tmp12 = tl.load(input_ptr1 + input_index1, valid_mask, eviction_policy='evict_last')

    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3

    tmp5 = 2 * (
        ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) *
        (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))) <=
         ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                  (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
         ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                  (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
         (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) +
                   (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
          (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))))
    )
    tmp6 = tmp5 + tmp2

    tmp7 = 2 * (
        ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) *
        (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))) <=
         ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                  (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) +
         ((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                  (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) *
         (((-1) + (((-1) + (kernel_size2 // 2)) * (((-1) + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) +
                   (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size2 // 2))))) <
          (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0))))))
    )
    tmp8 = tmp7 + tmp4

    tmp9 = kernel_size0
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp8

    tmp13 = block_indices % kernel_size1
    tmp14 = tmp11 == tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp14, tmp12, tmp15)

    tl.store(output_ptr0 + linear_index, tmp16, valid_mask)