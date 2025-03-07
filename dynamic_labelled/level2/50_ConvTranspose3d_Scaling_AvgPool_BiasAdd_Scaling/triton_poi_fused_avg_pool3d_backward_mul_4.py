# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_backward_mul_4poi_fused_avg_pool3d_backward_mul_4(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    kernel_size4, kernel_size5, kernel_size6, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    x_dim = block_indices % kernel_size0
    y_dim = (block_indices // kernel_size0) % kernel_size0
    z_dim = (block_indices // kernel_size1) % kernel_size2
    batch_index = block_indices // kernel_size3
    z_dim_2 = (block_indices // kernel_size6) % kernel_size2
    linear_index = block_indices

    offset_x = -batch_index
    offset_y = -((0 if (y_dim // 2) < 0 else y_dim // 2) * 
                 ((0 if (y_dim // 2) < 0 else y_dim // 2) <= (-1 + ((-1 + kernel_size5) * 
                 ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + (1 + (y_dim // 2)) * 
                 ((1 + (y_dim // 2)) < (-1 + kernel_size5))) + 
                 (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + 
                 (1 + (y_dim // 2)) * ((1 + (y_dim // 2)) < (-1 + kernel_size5))) * 
                 ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + 
                 (1 + (y_dim // 2)) * ((1 + (y_dim // 2)) < (-1 + kernel_size5)))) < 
                 (0 if (y_dim // 2) < 0 else y_dim // 2))))

    offset_z = ((0 if (x_dim // 2) < 0 else x_dim // 2) * 
                ((0 if (x_dim // 2) < 0 else x_dim // 2) <= (-1 + ((-1 + kernel_size5) * 
                ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + (1 + (x_dim // 2)) * 
                ((1 + (x_dim // 2)) < (-1 + kernel_size5))) + 
                (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + 
                (1 + (x_dim // 2)) * ((1 + (x_dim // 2)) < (-1 + kernel_size5))) * 
                ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + 
                (1 + (x_dim // 2)) * ((1 + (x_dim // 2)) < (-1 + kernel_size5)))) < 
                (0 if (x_dim // 2) < 0 else x_dim // 2))))

    offset_z2 = ((0 if (z_dim // 2) < 0 else z_dim // 2) * 
                 ((0 if (z_dim // 2) < 0 else z_dim // 2) <= (-1 + ((-1 + kernel_size4) * 
                 ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + (1 + (z_dim // 2)) * 
                 ((1 + (z_dim // 2)) < (-1 + kernel_size4))) + 
                 (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
                 (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4))) * 
                 ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
                 (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4)))) < 
                 (0 if (z_dim // 2) < 0 else z_dim // 2))))

    index = (offset_x + kernel_size4 * batch_index + 
             kernel_size5 * ((0 if (y_dim // 2) < 0 else y_dim // 2) * 
             ((0 if (y_dim // 2) < 0 else y_dim // 2) <= (-1 + ((-1 + kernel_size5) * 
             ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + (1 + (y_dim // 2)) * 
             ((1 + (y_dim // 2)) < (-1 + kernel_size5))) + 
             (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + 
             (1 + (y_dim // 2)) * ((1 + (y_dim // 2)) < (-1 + kernel_size5))) * 
             ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + 
             (1 + (y_dim // 2)) * ((1 + (y_dim // 2)) < (-1 + kernel_size5)))) < 
             (0 if (y_dim // 2) < 0 else y_dim // 2))) + 
             kernel_size5 * kernel_size5 * ((0 if (z_dim // 2) < 0 else z_dim // 2) * 
             ((0 if (z_dim // 2) < 0 else z_dim // 2) <= (-1 + ((-1 + kernel_size4) * 
             ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + (1 + (z_dim // 2)) * 
             ((1 + (z_dim // 2)) < (-1 + kernel_size4))) + 
             (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4))) * 
             ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4)))) < 
             (0 if (z_dim // 2) < 0 else z_dim // 2))) + 
             (-1) * batch_index * kernel_size5 * kernel_size5 + 
             (-2) * kernel_size5 * ((0 if (z_dim // 2) < 0 else z_dim // 2) * 
             ((0 if (z_dim // 2) < 0 else z_dim // 2) <= (-1 + ((-1 + kernel_size4) * 
             ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + (1 + (z_dim // 2)) * 
             ((1 + (z_dim // 2)) < (-1 + kernel_size4))) + 
             (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4))) * 
             ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4)))) < 
             (0 if (z_dim // 2) < 0 else z_dim // 2))) + 
             2 * kernel_size5 * batch_index + 
             kernel_size4 * batch_index * kernel_size5 * kernel_size5 + 
             (-2) * kernel_size4 * kernel_size5 * batch_index + 
             ((0 if (x_dim // 2) < 0 else x_dim // 2) * 
             ((0 if (x_dim // 2) < 0 else x_dim // 2) <= (-1 + ((-1 + kernel_size5) * 
             ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + (1 + (x_dim // 2)) * 
             ((1 + (x_dim // 2)) < (-1 + kernel_size5))) + 
             (-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + 
             (1 + (x_dim // 2)) * ((1 + (x_dim // 2)) < (-1 + kernel_size5))) * 
             ((-1 + ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + 
             (1 + (x_dim // 2)) * ((1 + (x_dim // 2)) < (-1 + kernel_size5)))) < 
             (0 if (x_dim // 2) < 0 else x_dim // 2))) + 
             ((0 if (z_dim // 2) < 0 else z_dim // 2) * 
             ((0 if (z_dim // 2) < 0 else z_dim // 2) <= (-1 + ((-1 + kernel_size4) * 
             ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + (1 + (z_dim // 2)) * 
             ((1 + (z_dim // 2)) < (-1 + kernel_size4))) + 
             (-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4))) * 
             ((-1 + ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim // 2))) + 
             (1 + (z_dim // 2)) * ((1 + (z_dim // 2)) < (-1 + kernel_size4)))) < 
             (0 if (z_dim // 2) < 0 else z_dim // 2)))))

    tmp0 = tl.load(input_ptr0 + index, valid_mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [BLOCK_SIZE])
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 / 8

    z_dim_2_condition = (0 if (z_dim_2 // 2) < 0 else z_dim_2 // 2)
    z_dim_2_limit = ((-1 + kernel_size4) * ((-1 + kernel_size4) <= (1 + (z_dim_2 // 2))) + 
                     (1 + (z_dim_2 // 2)) * ((1 + (z_dim_2 // 2)) < (-1 + kernel_size4)))
    z_dim_2_valid = z_dim_2_condition < z_dim_2_limit

    y_dim_condition = (0 if (y_dim // 2) < 0 else y_dim // 2)
    y_dim_limit = ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (y_dim // 2))) + 
                   (1 + (y_dim // 2)) * ((1 + (y_dim // 2)) < (-1 + kernel_size5)))
    y_dim_valid = y_dim_condition < y_dim_limit

    combined_valid_y_z = z_dim_2_valid & y_dim_valid

    x_dim_condition = (0 if (x_dim // 2) < 0 else x_dim // 2)
    x_dim_limit = ((-1 + kernel_size5) * ((-1 + kernel_size5) <= (1 + (x_dim // 2))) + 
                   (1 + (x_dim // 2)) * ((1 + (x_dim // 2)) < (-1 + kernel_size5)))
    x_dim_valid = x_dim_condition < x_dim_limit

    combined_valid = combined_valid_y_z & x_dim_valid

    result = tl.where(combined_valid, tmp4, 0.0)
    tl.store(output_ptr0 + linear_index, result, valid_mask)