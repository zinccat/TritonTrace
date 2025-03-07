# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_per_fused__softmax_add_convolution_mean_mul_tanh_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 1968624
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block = rindex
    x_block_1 = (xindex // 123039)
    x_block_2 = xindex % 123039
    x_block_3 = xindex % 3969
    x_block_4 = (xindex // 3969)
    x_flat_index = xindex

    # Load input data with masking
    input_data_0 = tl.load(in_ptr0 + (x_block_2 + (123039 * r_block) + (1968624 * x_block_1)), xmask, other=0.0)
    input_data_1 = tl.load(in_ptr1 + (r_block), None, eviction_policy='evict_last')
    input_data_2 = tl.load(in_ptr2 + (0))
    broadcast_input_data_2 = tl.broadcast_to(input_data_2, [XBLOCK, 1])

    # Compute intermediate values
    sum_input_data = input_data_0 + input_data_1
    broadcast_sum_input_data = tl.broadcast_to(sum_input_data, [XBLOCK, RBLOCK])
    masked_sum_input_data = tl.where(xmask, broadcast_sum_input_data, 0)
    sum_masked_data = tl.sum(masked_sum_input_data, 1)[:, None]
    mean_masked_data = sum_masked_data / 16.0
    adjusted_mean_data = mean_masked_data + broadcast_input_data_2
    centered_data = adjusted_mean_data - adjusted_mean_data
    exp_centered_data = tl.math.exp(centered_data)
    softmax_data = exp_centered_data / exp_centered_data
    tanh_softmax_data = tl.extra.cuda.libdevice.tanh(softmax_data)
    scaled_tanh_data = tanh_softmax_data * 2.0

    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_block_3 + (4000 * x_block_4)), softmax_data, xmask)
    tl.store(out_ptr0 + (x_flat_index), scaled_tanh_data, xmask)