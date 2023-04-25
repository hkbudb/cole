pub struct Configs {
    pub fanout: usize, // fanout of the MHT
    pub epsilon: i64, // upper error bound of the piecewise model
    pub dir_name: String, // directory name of the storage
    pub base_state_num: usize, // number of states in the in-memory level
    pub size_ratio: usize, // ratio of the number of states in between the two consecutive levels
}

impl Configs {
    pub fn new(fanout: usize, epsilon: i64, dir_name: String, base_state_num: usize, size_ratio: usize) -> Self {
        Self {
            fanout,
            epsilon,
            dir_name,
            base_state_num,
            size_ratio,
        }
    }

    pub fn max_num_of_states_in_a_run(&self, level_id: u32) -> usize {
        return self.base_state_num * self.size_ratio.pow(level_id);
    }
}

/*
Compute a recommended bitmap size for items_count items
and a fp_p rate of false positives.
fp_p obviously has to be within the ]0.0, 1.0[ range.
 */
pub fn compute_bitmap_size_in_bytes(items_count: usize, fp_p: f64) -> usize {
    assert!(items_count > 0);
    assert!(fp_p > 0.0 && fp_p < 1.0);
    // We're using ceil instead of round in order to get an error rate <= the desired.
    // Using round can result in significantly higher error rates.
    let num_slices = ((1.0 / fp_p).log2()).ceil() as u64;
    let slice_len_bits = (items_count as f64 / 2f64.ln()).ceil() as u64;
    let total_bits = num_slices * slice_len_bits;
    // round up to the next byte
    let buffer_bytes = ((total_bits + 7) / 8) as usize;
    buffer_bytes
}

#[cfg(test)]
mod tests {
    use rug::ops::Pow;
    use super::compute_bitmap_size_in_bytes;
    #[test]
    fn test_max_num_states() {
        let base = 10;
        let size_ratio = 2;
        for i in 0..5 {
            println!("level {} has maximum {} states", i, base * size_ratio.pow(i as u32));
        }
        println!("bloom filter size: {}", compute_bitmap_size_in_bytes(1000000, 0.1));
    }
}