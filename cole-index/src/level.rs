use utils::{config::Configs, OpenOptions, Write, Read, H256, types::compute_concatenate_hash};
use crate::run::{LevelRun, RunFilterSize};
use std::fmt::{Debug, Formatter, Error};
use std::thread;
// a level in cole
pub struct Level {
    pub level_id: u32, // id to identify the level
    pub run_vec: Vec<LevelRun>, // a vector to store the runs in the level
}

impl Level {
    // initiate a new level using level_id, the run_vec is empty
    pub fn new(level_id: u32) -> Self {
        Self {
            level_id,
            run_vec: vec![],
        }
    }
    // load the level from the file given level_id and the reference of configs
    pub fn load(level_id: u32, configs: &Configs) -> Self {
        let mut level = Self::new(level_id);
        let level_meta_file_name = level.level_meta_file_name(configs);
        match OpenOptions::new().read(true).open(&level_meta_file_name) {
            Ok(mut file) => {
                // read num_of_run from the file
                let mut num_of_run_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut num_of_run_bytes).unwrap();
                let num_of_run = u32::from_be_bytes(num_of_run_bytes) as usize;
                // read run_id from the file and load the run to the vector
                for _ in 0..num_of_run {
                    let mut run_id_bytes: [u8; 4] = [0x00; 4];
                    file.read_exact(&mut run_id_bytes).unwrap();
                    // deserialize the run_id
                    let run_id = u32::from_be_bytes(run_id_bytes);
                    // load the run 
                    let run = LevelRun::load(run_id, level_id, configs);
                    level.run_vec.push(run);
                }
            },
            Err(_) => {
            }
        }

        return level;
    }

    // persist the level, including the run_id and run's filter of each run in run_vec
    pub fn persist_level(&self, configs: &Configs) {
        let num_of_run = self.run_vec.len();
        // store the binary bytes of num_of_run to the output vector
        let mut v = (num_of_run as u32).to_be_bytes().to_vec();
        for i in 0..num_of_run {
            // get the run_id of the current run
            let run_id = self.run_vec[i].run_id;
            // store the binary of run_id to the output vector
            v.extend(&run_id.to_be_bytes());
            // persist the filter of the run if it exists
            self.run_vec[i].persist_filter(self.level_id, configs);
        }
        let level_meta_file_name = self.level_meta_file_name(configs);
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&level_meta_file_name).unwrap();
        // persist the output vector to the level's file
        file.write_all(&mut v ).unwrap();
    }

    pub fn level_meta_file_name(&self, configs: &Configs) -> String {
        format!("{}/{}.lv", &configs.dir_name, self.level_id)
    }
    // if the number of runs in the level is the same as the size ratio, the level is full
    pub fn level_reach_capacity(&self, configs: &Configs) -> bool {
        self.run_vec.len() == configs.size_ratio
    }
    // compute the digest of the level
    pub fn compute_digest(&self) -> H256 {
        let run_hash_vec: Vec<H256> = self.run_vec.iter().map(|run| run.compute_digest()).collect();
        compute_concatenate_hash(&run_hash_vec)
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
        for run in &self.run_vec {
            filter_size.add(&run.filter_cost());
        }
        return filter_size;
    }
}

impl Debug for Level {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
/*         for run in &self.run_vec {
            println!("{:?}", run);
        } */
        write!(f, "<Level Info> level_id: {}, num_of_runs: {}", self.level_id, self.run_vec.len())
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};
    use utils::config::Configs;
    use utils::pager::state_pager::InMemStateIterator;
    use utils::{H160, H256};
    use utils::types::{CompoundKey, StateValue};
    use crate::run::LevelRun;

    use super::Level;
    fn generate_run(level_id: u32, run_id: u32, n: usize, configs: &Configs, seed: u64) -> LevelRun{
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
    fn test_level() {
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

        let mut level = Level::new(level_id);
        for i in 0..2 {
            let run = generate_run(level_id, i, n, &configs, (i+1) as u64);
            level.run_vec.push(run);
        }
        level.persist_level(&configs);
        drop(level);
        let load_level = Level::load(level_id, &configs);
        println!("{:?}", load_level);
    }
}