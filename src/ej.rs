// Block size used by the tiling algoritms
const BLOCK_SIZE: usize = 16;
// Number of segments used by the segmented block transpose function
const NBR_SEGMENTS: usize = 4;
// recursively split data until the number of rows and columns is below this number
const RECURSIVE_LIMIT: usize = 128;

// Largest size for for using the direct approach
const SMALL_LEN: usize = 255;
// Largest size for using the tiled approach
const MEDIUM_LEN: usize = 1024 * 1024;

pub fn transpose<T: Copy>(input: &[T], output: &mut [T], input_width: usize, input_height: usize) {
    assert_eq!(input_width * input_height, input.len());
    assert_eq!(input_width * input_height, output.len());
    if input.len() <= SMALL_LEN {
        unsafe { transpose_small(input, output, input_width, input_height) };
    } else if input.len() <= MEDIUM_LEN {
        transpose_tiled(input, output, input_width, input_height);
    } else {
        transpose_recursive(
            input,
            output,
            0,
            input_height,
            0,
            input_width,
            input_width,
            input_height,
        );
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// Benchmarking shows that loop tiling isn't effective for small arrays.
unsafe fn transpose_small<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
unsafe fn transpose_block<T: Copy>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = start_x + inner_x;
            let y = start_y + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
// This function works as `transpose_block`, but also divides the loop into a number of segments. This makes it more cache fiendly for large sizes.
unsafe fn transpose_block_segmented<T: Copy>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    let height_per_div = block_height / NBR_SEGMENTS;
    for subblock in 0..NBR_SEGMENTS {
        for inner_x in 0..block_width {
            for inner_y in 0..height_per_div {
                let x = start_x + inner_x;
                let y = start_y + inner_y + subblock * height_per_div;

                let input_index = x + y * width;
                let output_index = y + x * height;

                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}

pub fn transpose_tiled<T: Copy>(
    input: &[T],
    output: &mut [T],
    input_width: usize,
    input_height: usize,
) {
    let x_block_count = input_width / BLOCK_SIZE;
    let y_block_count = input_height / BLOCK_SIZE;

    let remainder_x = input_width - x_block_count * BLOCK_SIZE;
    let remainder_y = input_height - y_block_count * BLOCK_SIZE;

    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    x_block * BLOCK_SIZE,
                    y_block * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    input_width - remainder_x,
                    y_block * BLOCK_SIZE,
                    remainder_x,
                    BLOCK_SIZE,
                );
            }
        }
    }

    //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    x_block * BLOCK_SIZE,
                    input_height - remainder_y,
                    BLOCK_SIZE,
                    remainder_y,
                );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    input_width - remainder_x,
                    input_height - remainder_y,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// This is a recursive algorithm that divides the array into smaller pieces, until they are small enough to
/// transpose directly without worrying about cache misses.
/// Once they are small enough, they are transposed using a tiling algorithm.
pub fn transpose_recursive<T: Copy>(
    input: &[T],
    output: &mut [T],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_columns: usize,
    total_rows: usize,
) {
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;
    if (nbr_rows <= RECURSIVE_LIMIT && nbr_cols <= RECURSIVE_LIMIT)
        || nbr_rows <= 2
        || nbr_cols <= 2
    {
        let x_block_count = nbr_cols / BLOCK_SIZE;
        let y_block_count = nbr_rows / BLOCK_SIZE;

        let remainder_x = nbr_cols - x_block_count * BLOCK_SIZE;
        let remainder_y = nbr_rows - y_block_count * BLOCK_SIZE;

        for y_block in 0..y_block_count {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block_segmented(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block * BLOCK_SIZE,
                        row_start + y_block * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE,
                    );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block_count * BLOCK_SIZE,
                        row_start + y_block * BLOCK_SIZE,
                        remainder_x,
                        BLOCK_SIZE,
                    );
                }
            }
        }

        //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
        if remainder_y > 0 {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block * BLOCK_SIZE,
                        row_start + y_block_count * BLOCK_SIZE,
                        BLOCK_SIZE,
                        remainder_y,
                    );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block_count * BLOCK_SIZE,
                        row_start + y_block_count * BLOCK_SIZE,
                        remainder_x,
                        remainder_y,
                    );
                }
            }
        }
    } else if nbr_rows >= nbr_cols {
        transpose_recursive(
            input,
            output,
            row_start,
            row_start + (nbr_rows / 2),
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
        transpose_recursive(
            input,
            output,
            row_start + (nbr_rows / 2),
            row_end,
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
    } else {
        transpose_recursive(
            input,
            output,
            row_start,
            row_end,
            col_start,
            col_start + (nbr_cols / 2),
            total_columns,
            total_rows,
        );
        transpose_recursive(
            input,
            output,
            row_start,
            row_end,
            col_start + (nbr_cols / 2),
            col_end,
            total_columns,
            total_rows,
        );
    }
}

use num_integer;
use strength_reduce::StrengthReducedUsize;

fn multiplicative_inverse(a: usize, n: usize) -> usize {
    // we're going to use a modified version extended euclidean algorithm
    // we only need half the output

    let mut t = 0;
    let mut t_new = 1;

    let mut r = n;
    let mut r_new = a;

    while r_new > 0 {
        let quotient = r / r_new;

        r = r - quotient * r_new;
        core::mem::swap(&mut r, &mut r_new);

        // t might go negative here, so we have to do a checked subtract
        // if it underflows, wrap it around to the other end of the modulo
        // IE, 3 - 4 mod 5  =  -1 mod 5  =  4
        let t_subtract = quotient * t_new;
        t = if t_subtract < t {
            t - t_subtract
        } else {
            n - (t_subtract - t) % n
        };
        core::mem::swap(&mut t, &mut t_new);
    }

    t
}

/// Transpose the input array in-place.
///
/// Given an input array of size input_width * input_height, representing flattened 2D data stored in row-major order,
/// transpose the rows and columns of that input array, in-place.
///
/// Despite being in-place, this algorithm requires max(width * height) in scratch space.
///
/// ``` ignore
/// // row-major order: the rows of our 2D array are contiguous,
/// // and the columns are strided
/// let original_array = vec![ 1, 2, 3,
///                            4, 5, 6];
/// let mut input_array = original_array.clone();
///
/// // Treat our 6-element array as a 2D 3x2 array, and transpose it to a 2x3 array
/// // transpose_inplace requires max(width, height) scratch space, which is in this case 3
/// let mut scratch = vec![0; 3];
/// transpose::transpose_inplace(&mut input_array, &mut scratch, 3, 2);
///
/// // The rows have become the columns, and the columns have become the rows
/// let expected_array =  vec![ 1, 4,
///                             2, 5,
///                             3, 6];
/// assert_eq!(input_array, expected_array);
///
/// // If we transpose it again, we should get our original data back.
/// transpose::transpose_inplace(&mut input_array, &mut scratch, 2, 3);
/// assert_eq!(original_array, input_array);
/// ```
///
/// # Panics
///
/// Panics if `input.len() != input_width * input_height` or if `output.len() != input_width * input_height`
pub fn transpose_inplace<T: Copy>(
    buffer: &mut [T],
    scratch: &mut [T],
    width: usize,
    height: usize,
) {
    assert_eq!(width * height, buffer.len());
    assert_eq!(core::cmp::max(width, height), scratch.len());

    let gcd = StrengthReducedUsize::new(num_integer::gcd(width, height));
    let a = StrengthReducedUsize::new(height / gcd);
    let b = StrengthReducedUsize::new(width / gcd);
    let a_inverse = multiplicative_inverse(a.get(), b.get());
    let strength_reduced_height = StrengthReducedUsize::new(height);

    let index_fn = |x, y| x + y * width;

    if gcd.get() > 1 {
        for x in 0..width {
            let column_offset = (x / b) % strength_reduced_height;
            let wrapping_point = height - column_offset;

            // wrapped rotation -- do the "right half" of the array, then the "left half"
            for y in 0..wrapping_point {
                scratch[y] = buffer[index_fn(x, y + column_offset)];
            }
            for y in wrapping_point..height {
                scratch[y] = buffer[index_fn(x, y + column_offset - height)];
            }

            // copy the data back into the column
            for y in 0..height {
                buffer[index_fn(x, y)] = scratch[y];
            }
        }
    }

    // Permute the rows
    {
        let row_scratch = &mut scratch[0..width];

        for (y, row) in buffer.chunks_exact_mut(width).enumerate() {
            for x in 0..width {
                let helper_val = if y <= height + x % gcd - gcd.get() {
                    x + y * (width - 1)
                } else {
                    x + y * (width - 1) + height
                };
                let (helper_div, helper_mod) = StrengthReducedUsize::div_rem(helper_val, gcd);

                let gather_x = (a_inverse * helper_div) % b + b.get() * helper_mod;
                row_scratch[x] = row[gather_x];
            }

            row.copy_from_slice(row_scratch);
        }
    }

    // Permute the columns
    for x in 0..width {
        let column_offset = x % strength_reduced_height;
        let wrapping_point = height - column_offset;

        // wrapped rotation -- do the "right half" of the array, then the "left half"
        for y in 0..wrapping_point {
            scratch[y] = buffer[index_fn(x, y + column_offset)];
        }
        for y in wrapping_point..height {
            scratch[y] = buffer[index_fn(x, y + column_offset - height)];
        }

        // Copy the data back to the buffer, but shuffle it as we do so
        for y in 0..height {
            let shuffled_y = (y * width - (y / a)) % strength_reduced_height;
            buffer[index_fn(x, y)] = scratch[shuffled_y];
        }
    }
}
