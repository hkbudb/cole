use cole_index::run::{LevelRun, RunFilterSize};
use utils::types::compute_concatenate_hash;
use utils::{config::Configs, OpenOptions, Write, Read, H256};
use std::thread::{JoinHandle, self};
use std::fmt::{Debug, Formatter, Error};
pub struct LevelGroup {
    pub run_vec: Vec<LevelRun>, // a vector of level runs
    pub thread_handle: Option<JoinHandle<LevelRun>>, // object related to the asynchronous merge thread
}

impl LevelGroup {
    pub fn new() -> Self {
        Self {
            run_vec: Vec::new(),
            thread_handle: None,
        }
    }
}

impl Debug for LevelGroup {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "run_vec len: {}\n", self.run_vec.len()).unwrap();
/*         for run in &self.run_vec {
            println!("run_id: {}, run: {:?}", run.run_id, run);
        } */
        write!(f, "thread is some: {}\n", self.thread_handle.is_some()).unwrap();
        Ok(())
    }
}

// an asynchronous level in Cole
pub struct AsyncLevel {
    pub level_id: u32, // id to identify the level
    pub run_groups: [LevelGroup; 2], // run groups of level runs
    pub write_group_flag: bool, // true means the first run group is the write group; false means the second run group is the write group
}

impl AsyncLevel {
    pub fn new(level_id: u32) -> Self {
        Self {
            level_id,
            run_groups: [LevelGroup::new(), LevelGroup::new()],
            write_group_flag: true,
        }
    }

    pub fn load(level_id: u32, configs: &Configs) -> Self {
        let mut level = Self::new(level_id);
        let level_meta_file_name = level.level_meta_file_name(configs);
        match OpenOptions::new().read(true).open(&level_meta_file_name) {
            Ok(mut file) => {
                // get write index of the write group
                let write_index = level.get_write_group_index();
                // read num_of_run of write group from the file
                let mut write_num_of_run_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut write_num_of_run_bytes).unwrap();
                let write_num_of_run = u32::from_be_bytes(write_num_of_run_bytes) as usize;
                // read run_id from the file and load the run to the vector
                for _ in 0..write_num_of_run {
                    let mut run_id_bytes: [u8; 4] = [0x00; 4];
                    file.read_exact(&mut run_id_bytes).unwrap();
                    // deserialize the run_id
                    let run_id = u32::from_be_bytes(run_id_bytes);
                    // load the run
                    let run = LevelRun::load(run_id, level_id, configs);
                    level.run_groups[write_index].run_vec.push(run);
                }

                // get merge index of the merge group
                let merge_index = level.get_merge_group_index();
                // read num_of_run of merge group from the file
                let mut merge_num_of_run_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut merge_num_of_run_bytes).unwrap();
                let merge_num_of_run = u32::from_be_bytes(merge_num_of_run_bytes) as usize;
                // read run_id from the file and load the run to the vector
                for _ in 0..merge_num_of_run {
                    let mut run_id_bytes: [u8; 4] = [0x00; 4];
                    file.read_exact(&mut run_id_bytes).unwrap();
                    // deserialize the run_id
                    let run_id = u32::from_be_bytes(run_id_bytes);
                    // load the run
                    let run = LevelRun::load(run_id, level_id, configs);
                    level.run_groups[merge_index].run_vec.push(run);
                }
            },
            Err(_) => {}
        }
        return level;
    }

    pub fn get_write_group_index(&self) -> usize {
        if self.write_group_flag == true {
            // the first run group is the write group
            return 0;
        } else {
            // the second run group is the write group
            return 1;
        }
    }

    pub fn get_merge_group_index(&self) -> usize {
        if self.write_group_flag == true {
            // the second run group is the merge group
            return 1;
        } else {
            // the first run group is the merge group
            return 0;
        }
    }

    // if the number of runs in the level is the same as the size ratio, the level is full
    pub fn level_write_group_reach_capacity(&self, configs: &Configs) -> bool {
        let write_group_index = self.get_write_group_index();
        self.run_groups[write_group_index].run_vec.len() >= configs.size_ratio
    }

    pub fn switch_group(&mut self) {
        // reverse the flag of write group
        if self.write_group_flag == true {
            self.write_group_flag = false;
        } else {
            self.write_group_flag = true;
        }
    }

    pub fn level_meta_file_name(&self, configs: &Configs) -> String {
        format!("{}/{}.lv", &configs.dir_name, self.level_id)
    }

    // persist the level, including the run_id and run's filter of each run in run_vec
    pub fn persist_level(&self, configs: &Configs) {
        // first persist the write group
        let write_index = self.get_write_group_index();
        let write_group_num_of_run = self.run_groups[write_index].run_vec.len();
        // store the binary bytes of num_of_run to the output vector
        let mut v = (write_group_num_of_run as u32).to_be_bytes().to_vec();
        for i in 0..write_group_num_of_run {
            // get the run_id of the current run
            let run_id = self.run_groups[write_index].run_vec[i].run_id;
            // store the run_id to the output vector
            v.extend(&run_id.to_be_bytes());
            // persist the filter of the run if it exists
            self.run_groups[write_index].run_vec[i].persist_filter(self.level_id, configs);
        }
        // second persist the merge group
        let merge_index = self.get_merge_group_index();
        let merge_group_num_of_run = self.run_groups[merge_index].run_vec.len();
        // store the binary bytes of num_of_run to the output vector
        v.extend((merge_group_num_of_run as u32).to_be_bytes());
        for i in 0..merge_group_num_of_run {
            // get the run_id of the current run
            let run_id = self.run_groups[merge_index].run_vec[i].run_id;
            // store the run_id to the output vector
            v.extend(&run_id.to_be_bytes());
            // persist the filter of the run if it exists
            self.run_groups[merge_index].run_vec[i].persist_filter(self.level_id, configs);
        }
        let level_meta_file_name = self.level_meta_file_name(configs);
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&level_meta_file_name).unwrap();
        // persist the output vector to the level's file
        file.write_all(&mut v).unwrap();
    }

    // compute the digest of the level
    pub fn compute_digest(&self) -> H256 {
        // write_group_run_hash_vec
        let write_index = self.get_write_group_index();
        let write_run_hash_vec: Vec<H256> = self.run_groups[write_index].run_vec.iter().map(|run| run.compute_digest()).collect();
        // merge_group_run_hash_vec
        let merge_index = self.get_merge_group_index();
        let merge_run_hash_vec: Vec<H256> = self.run_groups[merge_index].run_vec.iter().map(|run| run.compute_digest()).collect();
        let mut total_run_hash_vec = vec![];
        total_run_hash_vec.extend(write_run_hash_vec);
        total_run_hash_vec.extend(merge_run_hash_vec);
        compute_concatenate_hash(&total_run_hash_vec)
    }

    pub fn remove_run_files(run_id_vec: Vec<u32>, level_id: u32, dir_name: &str) {
        for run_id in run_id_vec {
            let state_file_name = LevelRun::file_name(run_id, level_id, dir_name, "s");
            let model_file_name = LevelRun::file_name(run_id, level_id, dir_name, "m");
            let mht_file_name = LevelRun::file_name(run_id, level_id, dir_name, "h");
            thread::spawn(move || {
                std::fs::remove_file(&state_file_name).unwrap();
            });
            thread::spawn(move || {
                std::fs::remove_file(&model_file_name).unwrap();
            });
            thread::spawn(move || {
                std::fs::remove_file(&mht_file_name).unwrap();
            });
        }
    }

    // compute filter cost
    pub fn filter_cost(&self) -> RunFilterSize {
        let mut filter_size = RunFilterSize::new(0);
        for run in &self.run_groups[0].run_vec {
            filter_size.add(&run.filter_cost());
        }
        for run in &self.run_groups[1].run_vec {
            filter_size.add(&run.filter_cost());
        }
        return filter_size;
    }
}

impl Debug for AsyncLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let write_index = self.get_write_group_index();
        let merge_index = self.get_merge_group_index();
        write!(f, "write index: {}, merge index: {}, flag: {}\n", write_index, merge_index, self.write_group_flag).unwrap();
        write!(f, "write group: {:?}\n", self.run_groups[write_index]).unwrap();
        write!(f, "merge group: {:?}\n", self.run_groups[merge_index]).unwrap();
        write!(f, "<Level Info> level_id: {}", self.level_id)
    }
}
#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};
    use utils::config::Configs;
    use utils::pager::state_pager::InMemStateIterator;
    use utils::{H160, H256};
    use utils::types::{CompoundKey, StateValue};
    use cole_index::run::LevelRun;
    use super::*;

    fn generate_run(level_id: u32, run_id: u32, n: usize, configs: &Configs, seed: u64) -> LevelRun {
        let mut iters = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let value = H256::random_using(&mut rng);
            let key = CompoundKey::new(acc_addr.into(), state_addr.into(), version);
            let value = StateValue(value);
            state_vec.push((key, value));
        }
        let min_key = CompoundKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into(), 0);
        let max_key = CompoundKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX);
        state_vec.push((min_key, StateValue(H256::default())));
        state_vec.push((max_key, StateValue(H256::default())));
        state_vec.sort();
        let it = InMemStateIterator::create(state_vec);
        iters.push(it);
        
        let run = LevelRun::construct_run_by_in_memory_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, 1);
        return run;
    }

    #[test]
    fn test_async_level() {
        let level_id = 0;
        let epsilon = 46;
        let fanout = 2;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let n = 100;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n,
            size_ratio: 10,
        };

        let mut level = AsyncLevel::new(level_id);
        level.switch_group();
        let write_index = level.get_write_group_index();
        let merge_index = level.get_merge_group_index();
        for i in 0..2 {
            let run = generate_run(level_id, i, n, &configs, (i+1) as u64);
            level.run_groups[write_index].run_vec.push(run);
        }

        let run = generate_run(level_id, 2, n, &configs, 3);
        level.run_groups[merge_index].run_vec.push(run);

        level.persist_level(&configs);
        println!("before level: {:?}", level);
        drop(level);
        let load_level = AsyncLevel::load(level_id, &configs);
        println!("{:?}", load_level);
    }
}