pub mod async_level;
use async_level::AsyncLevel;
use cole_index::{in_memory_mbtree::InMemoryMBTree, run::{LevelRun, reconstruct_run_proof, RunFilterSize}, RunProofOrHash, MemCost};
use merkle_btree_storage::{traits::BPlusTreeNodeIO, RangeProof as MBTreeRangeProof};
use std::thread::JoinHandle;
use utils::{config::Configs, types::{CompoundKeyValue, StateValue, compute_concatenate_hash, AddrKey, CompoundKey}, pager::state_pager::{InMemStateIterator, StatePageReader, StateIterator}, OpenOptions, Write, Read, H256, cacher::CacheManager};
use std::fmt::{Debug, Formatter, Error};
use serde::{Serialize, Deserialize};
pub const EVAL_STORAGE_AFTER_DROP: bool = true;
pub struct InMemGroup {
    pub mem_mht: InMemoryMBTree, // in-mem MB-tree
    pub thread_handle: Option<JoinHandle<LevelRun>>, // object related to the asynchronous merge thread,
}

impl Debug for InMemGroup {
    fn fmt(&self, _: &mut Formatter<'_>) -> Result<(), Error> {
        println!("mem mht: {:?}", self.mem_mht);
        println!("thread is some: {}", self.thread_handle.is_some());
        Ok(())
    }
}

impl InMemGroup {
    pub fn new(fanout: usize) -> Self {
        Self {
            mem_mht: InMemoryMBTree::new(fanout),
            thread_handle: None,
        }
    }
    pub fn clear(&mut self) {
        self.mem_mht.clear();
        self.thread_handle = None;
    }
}

/* COLEStart consists of:
    (i) a reference of configs that include params
    (ii) two InMemGroups
    (iii) a write_group_flag to determine whether the first group of mem_mht is the writing group or the second is
    (iv) a vector of levels that stores each level's LevelRuns
 */
pub struct ColeStar<'a> {
    pub configs: &'a Configs,
    pub in_mem_group: [InMemGroup; 2],
    pub in_mem_write_group_flag: bool,
    pub levels: Vec<AsyncLevel>,
    pub run_id_cnt: u32, // this helps generate a new run_id
    pub cache_manager: CacheManager,
}

impl<'a> ColeStar<'a> {
    // create a new index using configs,
    pub fn new(configs: &'a Configs) -> Self {
        Self {
            configs,
            in_mem_group: [InMemGroup::new(configs.fanout), InMemGroup::new(configs.fanout)],
            in_mem_write_group_flag: true,
            levels: Vec::new(), // empty levels' vector
            run_id_cnt: 0, // initiate the counter to be 0
            cache_manager: CacheManager::new(),
        }
    }

    fn get_meta(&mut self) -> usize {
        let path = self.get_tree_meta_path();
        let mut file = OpenOptions::new().create(true).read(true).write(true).open(&path).unwrap();
        // read level len
        let mut level_len_bytes = [0u8; 4];
        let mut level_len: u32 = 0;
        match file.read_exact(&mut level_len_bytes) {
            Ok(_) => {
                level_len = u32::from_be_bytes(level_len_bytes);
            },
            Err(_) => {}
        }
        // read run_id_cnt
        let mut run_id_cnt_bytes = [0u8; 4];
        let mut run_id_cnt: u32 = 0;
        match file.read_exact(&mut run_id_cnt_bytes) {
            Ok(_) => {
                run_id_cnt = u32::from_be_bytes(run_id_cnt_bytes);
            },
            Err(_) => {}
        }
        self.run_id_cnt = run_id_cnt;
        // read mem mht in the write group
        let mut write_mht_len_bytes = [0u8; 4];
        let mut write_mht_len = 0;
        match file.read_exact(&mut write_mht_len_bytes) {
            Ok(_) => {
                write_mht_len = u32::from_be_bytes(write_mht_len_bytes);
            },
            Err(_) => {}
        }
        let mut write_mht_bytes = vec![0u8; write_mht_len as usize];
        let write_index = self.get_write_in_mem_group_index();
        match file.read_exact(&mut write_mht_bytes) {
            Ok(_) => {
                self.in_mem_group[write_index].mem_mht = bincode::deserialize(&write_mht_bytes).unwrap();
            },
            Err(_) => {},
        }
        // read mem mht in the merge group
        let mut merge_mht_len_bytes = [0u8; 4];
        let mut merge_mht_len = 0;
        match file.read_exact(&mut merge_mht_len_bytes) {
            Ok(_) => {
                merge_mht_len = u32::from_be_bytes(merge_mht_len_bytes);
            },
            Err(_) => {}
        }
        let mut merge_mht_bytes = vec![0u8; merge_mht_len as usize];
        let merge_index = self.get_merge_in_mem_group_index();
        match file.read_exact(&mut merge_mht_bytes) {
            Ok(_) => {
                self.in_mem_group[merge_index].mem_mht = bincode::deserialize(&merge_mht_bytes).unwrap();
            },
            Err(_) => {},
        }
        return level_len as usize;
    }

    // load a new index using configs
    pub fn load(configs: &'a Configs) -> Self {
        let mut ret = Self::new(configs);
        let level_len = ret.get_meta();
        // load levels
        for i in 0..level_len {
            let level = AsyncLevel::load(i as u32, configs);
            ret.levels.push(level);
        }
        return ret;
    }

    fn new_run_id(&mut self) -> u32 {
        // increment the run_id and return it
        self.run_id_cnt += 1;
        return self.run_id_cnt;
    }

    pub fn get_write_in_mem_group_index(&self) -> usize {
        if self.in_mem_write_group_flag == true {
            // the first in_mem group is the write group
            0
        } else {
            // the second in_mem group is the write group
            1
        }
    }

    pub fn get_merge_in_mem_group_index(&self) -> usize {
        if self.in_mem_write_group_flag == true {
            // the second in_mem group is the merge group
            1
        } else {
            // the first in_mem group is the merge group
            0
        }
    }

    pub fn insert(&mut self, state: CompoundKeyValue) {
        // compute the in-memory threshold
        let in_mem_thres = (self.configs.base_state_num as f64 * 0.5) as usize;
        // get the write in_mem group index
        let write_index = self.get_write_in_mem_group_index();
        // insert the state to the mb-tree of write group
        let mb_tree_ref = &mut self.in_mem_group[write_index].mem_mht;
        merkle_btree_storage::insert(mb_tree_ref, state.key, state.value);
        // check wheither the write group mb-tree is full
        if mb_tree_ref.key_num as usize == in_mem_thres {
            // get the merge group index
            let merge_index = self.get_merge_in_mem_group_index();
            let level_id = 0; // the first on-disk level's id is 0
            // check if the merge group has thread
            if let Some(handle) = self.in_mem_group[merge_index].thread_handle.take() {
                // get the merged new run
                let new_run = handle.join().unwrap();
                // add the new run to the first disk level's write group
                self.add_run_to_level_write_group(level_id, new_run);
                // clear the mb-tree of the merge group and set thread_handle to be None
                self.in_mem_group[merge_index].clear();
            }
            // switch the write group and merge group
            self.switch_in_mem_group();
            // get the updated merge index
            let merge_index = self.get_merge_in_mem_group_index();
            // the in-memory mb-tree is full, the data should be merged to the run in the disk-level
            let key_values = self.in_mem_group[merge_index].mem_mht.load_all_key_values();
            // prepare for the thread input
            let (new_run_id, dir_name, epsilon, fanout, max_num_of_states, level_num_of_run, size_ratio) = self.prepare_thread_input(level_id);
            // create a merge thread
            let handle = std::thread::spawn(move|| {
                let iter = InMemStateIterator::create(key_values);
                let inputs = vec![iter];
                let run = LevelRun::construct_run_by_in_memory_merge(inputs, new_run_id, level_id, &dir_name, epsilon, fanout, max_num_of_states, level_num_of_run, size_ratio);
                return run;
            });
            // assign the thread_handle to the merge group
            self.in_mem_group[merge_index].thread_handle = Some(handle);
            // check and merge the disk levels
            self.check_and_merge();
        }
    }

    fn check_and_merge(&mut self) {
        let mut level_id = 0; // start from 0 disk level
        // iteratively check each level's write group is full or not
        while level_id < self.levels.len() {
            if self.levels[level_id].level_write_group_reach_capacity(&self.configs) {
                // get the merge group's index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // get the next level id
                let next_level_id = level_id + 1;
                // check whether the merge group has thread_handle
                if let Some(handle) = self.levels[level_id].run_groups[merge_index].thread_handle.take() {
                    // get the merged new run
                    let new_run = handle.join().unwrap();
                    // add the new run to the next level
                    self.add_run_to_level_write_group(next_level_id as u32, new_run);
                    let merge_group_ref = &mut self.levels[level_id as usize].run_groups[merge_index];
                    // set the thread_handle to be None
                    merge_group_ref.thread_handle = None;
                    // remove all the runs in the merge group in levels[level_id]
                    let run_id_vec: Vec<u32> = merge_group_ref.run_vec.drain(..).map(|run| run.run_id).collect();
                    // remove the merged files in level_id by using multi-threads; note that we do not need to wait for the ending of the thread.
                    AsyncLevel::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
                }
                // switch the write group and merge group
                self.levels[level_id].switch_group();
                // get the updated merge group index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // prepare for run_ids of the merged runs, which will be used during the background merge thread
                let merge_group_run_id_vec: Vec<u32> = self.levels[level_id].run_groups[merge_index].run_vec.iter().map(|run| run.run_id).collect();
                // prepare for the input parameters of the merging thread
                let (new_run_id, dir_name, epsilon, fanout, max_num_of_states, level_num_of_run, size_ratio) = self.prepare_thread_input(next_level_id as u32);
                let handle = std::thread::spawn(move|| {
                        let inputs: Vec<StateIterator> = merge_group_run_id_vec.into_iter().map(|merge_run_id| {
                        let state_file_name = LevelRun::file_name(merge_run_id, level_id as u32, &dir_name, "s");
                        let state_reader = StatePageReader::load(&state_file_name);
                        state_reader.to_state_iter()
                    }).collect();
                    let run = LevelRun::construct_run_by_merge(inputs, new_run_id, next_level_id as u32, &dir_name, epsilon, fanout, max_num_of_states, level_num_of_run, size_ratio);
                    return run;
                });
                // assign the merge thread handle to level_id's merge group
                self.levels[level_id].run_groups[merge_index].thread_handle = Some(handle);
                level_id += 1;
            } else {
                break;
            }
        }
    }

    pub fn search_latest_state_value(&mut self, addr_key: AddrKey) -> Option<StateValue> {
        // compute the boundary compound key
        let upper_key = CompoundKey {
            addr: addr_key,
            version: u32::MAX,
        };
        // search the write-group in-mem tree
        let write_index = self.get_write_in_mem_group_index();
        let write_tree = &mut self.in_mem_group[write_index].mem_mht;
        match merkle_btree_storage::search_without_proof(write_tree, upper_key) {
            Some((read_key, read_v)) => {
                if read_key.addr == addr_key {
                    // matches the addresses and should be the latest value since latest value should be in the upper levels
                    return Some(read_v);
                }
            },
            None => {},
        }
        // search the merge-group in-mem tree 
        let merge_index = self.get_merge_in_mem_group_index();
        let merge_tree = &mut self.in_mem_group[merge_index].mem_mht;
        match merkle_btree_storage::search_without_proof(merge_tree, upper_key) {
            Some((read_key, read_v)) => {
                if read_key.addr == addr_key {
                    // matches the addresses and should be the latest value since latest value should be in the upper levels
                    return Some(read_v);
                }
            },
            None => {},
        }

        // search other levels on the disk
        for level in &mut self.levels {
            // first search the write group
            let write_index = level.get_write_group_index();
            for run in &mut level.run_groups[write_index].run_vec {
                let res = run.search_run(addr_key, &self.configs, &mut self.cache_manager);
                if res.is_some() {
                    let res = res.unwrap();
                    return Some(res.value);
                }
            }
            // then search the merge group
            let merge_index = level.get_merge_group_index();
            for run in &mut level.run_groups[merge_index].run_vec {
                let res = run.search_run(addr_key, &self.configs, &mut self.cache_manager);
                if res.is_some() {
                    let res = res.unwrap();
                    return Some(res.value);
                }
            }
        }
        return None;
    }
    
    // generate the range proof given the addr_key and two version ranges
    pub fn search_with_proof(&mut self, addr_key: AddrKey, low_version: u32, upper_version: u32) -> ColeStarProof {
        let mut proof = ColeStarProof::new();
        // generate the two compound keys
        let low_key = CompoundKey::new_with_addr_key(addr_key, low_version);
        let upper_key = CompoundKey::new_with_addr_key(addr_key, upper_version);
        let mut rest_is_hash = false;
        // search the write group of the in-memory tree
        let write_index = self.get_write_in_mem_group_index();
        let (r, p) = merkle_btree_storage::get_range_proof(&mut self.in_mem_group[write_index].mem_mht, low_key, upper_key);
        if r.is_some() {
            // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
            // there is no need to prove_range the run
            let left_most_result = r.as_ref().unwrap()[0].0;
            let result_version = left_most_result.version;
            if result_version < low_version {
                rest_is_hash = true;
            }
        }
        proof.in_mem_level.set_write_group(r, p);
        // search the merge group of the in-memory tree
        let merge_index = self.get_merge_in_mem_group_index();
        let (r, p) = merkle_btree_storage::get_range_proof(&mut self.in_mem_group[merge_index].mem_mht, low_key, upper_key);
        if r.is_some() {
            // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
            // there is no need to prove_range the run
            let left_most_result = r.as_ref().unwrap()[0].0;
            let result_version = left_most_result.version;
            if result_version < low_version {
                rest_is_hash = true;
            }
        }
        proof.in_mem_level.set_merge_group(r, p);

        let mut disk_level_vec = Vec::new();
        // search the runs in all disk levels
        
        for level in &mut self.levels {
            let mut level_proof = ColeStarLevelProof::new();
            // write group
            let write_index = level.get_write_group_index();
            for run in &mut level.run_groups[write_index].run_vec {
                // decide to add the run's proof or the run's digest
                if rest_is_hash == false {
                    let (r, p) = run.prove_range(addr_key, low_version, upper_version, &self.configs, &mut self.cache_manager);
                    if r.is_some() {
                        // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
                        // there is no need to prove_range the run
                        let left_most_result = r.as_ref().unwrap()[1];
                        let result_version = left_most_result.key.version;
                        if result_version < low_version {
                            rest_is_hash = true;
                        }
                    }
                    level_proof.push_to_write_group(r, RunProofOrHash::Proof(p));
                } else {
                    level_proof.push_to_write_group(None, RunProofOrHash::Hash(run.digest));
                }
            }
            // merge group
            let merge_index = level.get_merge_group_index();
            for run in &mut level.run_groups[merge_index].run_vec {
                // decide to add the run's proof or the run's digest
                if rest_is_hash == false {
                    let (r, p) = run.prove_range(addr_key, low_version, upper_version, &self.configs, &mut self.cache_manager);
                    if r.is_some() {
                        // check if the left_most version is smaller than the low_version, it means all the digests of the rest of the runs should be added to the proof
                        // there is no need to prove_range the run
                        let left_most_result = r.as_ref().unwrap()[1];
                        let result_version = left_most_result.key.version;
                        if result_version < low_version {
                            rest_is_hash = true;
                        }
                    }
                    level_proof.push_to_write_group(r, RunProofOrHash::Proof(p));
                } else {
                    level_proof.push_to_write_group(None, RunProofOrHash::Hash(run.digest));
                }
            }
            // add the level_proof to disk_level_vec
            disk_level_vec.push(level_proof);
        }
        // add the disk level's proofs to the proof
        proof.disk_level = disk_level_vec;
        return proof;
    }

    fn switch_in_mem_group(&mut self) {
        // reverse the flag of write group
        if self.in_mem_write_group_flag == true {
            self.in_mem_write_group_flag = false;
        } else {
            self.in_mem_write_group_flag = true;
        }
    }

    // add the new run to the level's write group with level_id
    fn add_run_to_level_write_group(&mut self, level_id: u32, new_run: LevelRun) {
        match self.levels.get_mut(level_id as usize) {
            Some(level_ref) => {
                let write_index = level_ref.get_write_group_index();
                level_ref.run_groups[write_index].run_vec.insert(0, new_run); // always insert the new run to the front, so that the latest states are at the front of the level
            },
            None => {
                let mut level = AsyncLevel::new(level_id); // the level with level_id does not exist, so create a new one
                let write_index = level.get_write_group_index();
                level.run_groups[write_index].run_vec.insert(0, new_run);
                self.levels.push(level); // push the new level to the level vector
            }
        }
    }

    fn prepare_thread_input(&mut self, level_id: u32) -> (u32, String, i64, usize, usize, usize, usize) {
        let new_run_id = self.new_run_id();
        let dir_name = self.configs.dir_name.clone();
        let epsilon = self.configs.epsilon;
        let fanout = self.configs.fanout;
        let size_ratio = self.configs.size_ratio;
        let max_num_of_states = self.configs.max_num_of_states_in_a_run(level_id);
        let level_num_of_run = match self.levels.get(level_id as usize) {
            Some(level) => {
                let write_index = level.get_write_group_index();
                level.run_groups[write_index].run_vec.len()
            },
            None => 0
        };
        (new_run_id, dir_name, epsilon, fanout, max_num_of_states, level_num_of_run, size_ratio)
    }

    fn get_tree_meta_path(&self) -> String {
        format!("{}/mht", &self.configs.dir_name)
    }

    fn update_manifest(&self) {
        // first persist all levels
        for level in &self.levels {
            level.persist_level(&self.configs);
        }
        // persist level len
        let level_len = self.levels.len() as u32;
        let mut bytes = level_len.to_be_bytes().to_vec();
        // persist run_id_cnt
        let run_id_cnt = self.run_id_cnt;
        bytes.extend(run_id_cnt.to_be_bytes());
        // serialize write in_mem mht
        let write_index = self.get_write_in_mem_group_index();
        let write_mht_bytes = bincode::serialize(&self.in_mem_group[write_index].mem_mht).unwrap();
        let write_mht_len = write_mht_bytes.len() as u32;
        bytes.extend(write_mht_len.to_be_bytes());
        bytes.extend(&write_mht_bytes);
        // serialize merge in_mem mht
        let merge_index = self.get_merge_in_mem_group_index();
        let merge_mht_bytes = bincode::serialize(&self.in_mem_group[merge_index].mem_mht).unwrap();
        let merge_mht_len = merge_mht_bytes.len() as u32;
        bytes.extend(merge_mht_len.to_be_bytes());
        bytes.extend(&merge_mht_bytes);
        // persist the bytes to the manifest file
        let path = self.get_tree_meta_path();
        let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&path).unwrap();
        file.write_all(&mut bytes).unwrap();
    }

    // compute the digest of COLE*
    pub fn compute_digest(&self) -> H256 {
        let mut hash_vec = vec![];
        // collect the write and merge group of in_mem_mht
        let write_index = self.get_write_in_mem_group_index();
        hash_vec.push(self.in_mem_group[write_index].mem_mht.get_root_hash());
        let merge_index = self.get_merge_in_mem_group_index();
        hash_vec.push(self.in_mem_group[merge_index].mem_mht.get_root_hash());
        let disk_hash_vec: Vec<H256> = self.levels.iter().map(|level| level.compute_digest()).collect();
        hash_vec.extend(&disk_hash_vec);
        compute_concatenate_hash(&hash_vec)
    }

    // compute filter cost
    pub fn filter_cost(&self) -> RunFilterSize {
        let mut filter_size = RunFilterSize::new(0);
        for level in &self.levels {
            filter_size.add(&level.filter_cost());
        }
        return filter_size;
    }

    pub fn memory_cost(&self) -> MemCost {
        let filter_size = self.filter_cost().filter_size;
        let (state_cache_size, model_cache_size, mht_cache_size) = self.cache_manager.compute_cacher_size();
        MemCost::new(state_cache_size, model_cache_size, mht_cache_size, filter_size)
    }
}

pub fn verify_and_collect_result(addr_key: AddrKey, low_version: u32, upper_version: u32, root_hash: H256, proof: &ColeStarProof, fanout: usize) -> (bool, Option<Vec<CompoundKeyValue>>) {
    let mut level_roots = Vec::<H256>::new();
    // first reconstruct the write group of the in-mem tree
    let low_key = CompoundKey::new_with_addr_key(addr_key, low_version);
    let upper_key = CompoundKey::new_with_addr_key(addr_key, upper_version);
    // retrieve write group result and proof
    let write_group_result = &proof.in_mem_level.write_group.0;
    let write_group_proof = &proof.in_mem_level.write_group.1;
    let h = merkle_btree_storage::reconstruct_range_proof(low_key, upper_key, write_group_result, write_group_proof);
    level_roots.push(h);
    let mut merge_result: Vec<CompoundKeyValue> = vec![];
    let mut rest_is_hash = false;
    if write_group_result.is_some() {
        let left_most_result = write_group_result.as_ref().unwrap()[0].0;
        let result_version = left_most_result.version;
        if result_version < low_version {
            rest_is_hash = true;
        }
        let r: Vec<CompoundKeyValue> = write_group_result.as_ref().unwrap().iter().map(|(k, v)| {
            CompoundKeyValue::new_with_compound_key(*k, *v)
        }).collect();
        merge_result.extend(r);
    }
    // then reconstruct merge group of the in-mem tree
    // retrieve merge group result and proof
    let merge_group_result = &proof.in_mem_level.merge_group.0;
    let merge_group_proof = &proof.in_mem_level.merge_group.1;
    let h = merkle_btree_storage::reconstruct_range_proof(low_key, upper_key, merge_group_result, merge_group_proof);
    level_roots.push(h);
    if merge_group_result.is_some() {
        let left_most_result = merge_group_result.as_ref().unwrap()[0].0;
        let result_version = left_most_result.version;
        if result_version < low_version {
            rest_is_hash = true;
        }
        let r: Vec<CompoundKeyValue> = merge_group_result.as_ref().unwrap().iter().map(|(k, v)| {
            CompoundKeyValue::new_with_compound_key(*k, *v)
        }).collect();
        merge_result.extend(r);
    }

    for level in &proof.disk_level {
        let mut level_write_h_vec: Vec<H256> = Vec::new();
        let mut level_merge_h_vec: Vec<H256> = Vec::new();
        let write_runs = &level.write_group;
        let merge_runs = &level.merge_group;
        for run in write_runs {
            let r = &run.0;
            let p = &run.1;
            match p {
                RunProofOrHash::Hash(h) => {
                    if rest_is_hash == false {
                        // in-complete result, return false
                        return (false, None);
                    }
                    level_write_h_vec.push(*h);
                },
                RunProofOrHash::Proof(proof) => {
                    if rest_is_hash == true {
                        // in-complete result, return false
                        return (false, None);
                    }
                    let (_, h) = reconstruct_run_proof(addr_key, low_version, upper_version, r, proof, fanout);
                    level_write_h_vec.push(h);
                }
            }
            if r.is_some() {
                let left_most_result = r.as_ref().unwrap()[1];
                let result_version = left_most_result.key.version;
                if result_version < low_version {
                    rest_is_hash = true;
                }
                merge_result.extend(r.as_ref().unwrap());
            }
        }

        for run in merge_runs {
            let r = &run.0;
            let p = &run.1;
            match p {
                RunProofOrHash::Hash(h) => {
                    if rest_is_hash == false {
                        // in-complete result, return false
                        return (false, None);
                    }
                    level_merge_h_vec.push(*h);
                },
                RunProofOrHash::Proof(proof) => {
                    if rest_is_hash == true {
                        // in-complete result, return false
                        return (false, None);
                    }
                    let (_, h) = reconstruct_run_proof(addr_key, low_version, upper_version, r, proof, fanout);
                    level_merge_h_vec.push(h);
                }
            }
            if r.is_some() {
                let left_most_result = r.as_ref().unwrap()[1];
                let result_version = left_most_result.key.version;
                if result_version < low_version {
                    rest_is_hash = true;
                }
                merge_result.extend(r.as_ref().unwrap());
            }
        }
        let mut total_run_hash_vec = vec![];
        total_run_hash_vec.extend(level_write_h_vec);
        total_run_hash_vec.extend(level_merge_h_vec);
        let level_h = compute_concatenate_hash(&total_run_hash_vec);
        level_roots.push(level_h);
    }

    let reconstruct_root = compute_concatenate_hash(&level_roots);
    if reconstruct_root != root_hash {
        return (false, None);
    }
    merge_result.sort_by(|a, b| a.key.partial_cmp(&b.key).unwrap());
    merge_result = merge_result.into_iter().filter(|r| r.key >= low_key && r.key <= upper_key).collect();
    return (true, Some(merge_result));
}

impl<'a> Drop for ColeStar<'a> {
    fn drop(&mut self) {
        if EVAL_STORAGE_AFTER_DROP == false {
            // first handle the in mem level's merge thread
            let merge_index = self.get_merge_in_mem_group_index();
            let mut level_id = 0;
            if let Some(handle) = self.in_mem_group[merge_index].thread_handle.take() {
                // get the merged new run
                let new_run = handle.join().unwrap();
                // add the new run to the first disk level's write group
                self.add_run_to_level_write_group(level_id as u32, new_run);
                // clear the mb-tree of the merge group and set thread_handle to be None
                self.in_mem_group[merge_index].clear();
            }
            // then handle the disk level's merge thread, note that here we do not need to care about whether the write group is full or not after committing the merge thread
            while level_id < self.levels.len() {
                // get the merge group's index
                let merge_index = self.levels[level_id].get_merge_group_index();
                // get the next level id
                let next_level_id = level_id + 1;
                // check whether the merge group has thread_handle
                if let Some(handle) = self.levels[level_id].run_groups[merge_index].thread_handle.take() {
                    // get the merged new run
                    let new_run = handle.join().unwrap();
                    // add the new run to the next level
                    self.add_run_to_level_write_group(next_level_id as u32, new_run);
                    let merge_group_ref = &mut self.levels[level_id as usize].run_groups[merge_index];
                    // set the thread_handle to be None
                    merge_group_ref.thread_handle = None;
                    // remove all the runs in the merge group in levels[level_id]
                    let run_id_vec: Vec<u32> = merge_group_ref.run_vec.drain(..).map(|run| run.run_id).collect();
                    // remove the merged files in level_id by using multi-threads; note that we do not need to wait for the ending of the thread.
                    AsyncLevel::remove_run_files(run_id_vec, level_id as u32, &self.configs.dir_name);
                }
                level_id += 1;
            }
        }
        // lastly persist the manifest
        self.update_manifest();
    }
}

impl<'a> Debug for ColeStar<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let write_index = self.get_write_in_mem_group_index();
        let merge_index = self.get_merge_in_mem_group_index();
        write!(f, "flag: {}, write_index: {}, merge_index: {}\n", self.in_mem_write_group_flag, write_index, merge_index).unwrap();
        write!(f, "write mem: {:?}\n", self.in_mem_group[write_index].mem_mht.get_key_num()).unwrap();
        write!(f, "merge mem: {:?}\n", self.in_mem_group[merge_index].mem_mht.get_key_num()).unwrap();
        write!(f, "run_id_cnt: {}\n", self.run_id_cnt).unwrap();
        for (i, level) in self.levels.iter().enumerate() {
            write!(f, "level {}: {:?}\n", i, level).unwrap();
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColeStarProof {
    pub in_mem_level: ColeStarInMemProof,
    pub disk_level: Vec<ColeStarLevelProof>,
}

impl ColeStarProof {
    // initiate the cole-start's proof
    pub fn new() -> Self {
        Self {
            in_mem_level: ColeStarInMemProof::new(),
            disk_level: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColeStarInMemProof {
    pub write_group: (Option<Vec<(CompoundKey, StateValue)>>, MBTreeRangeProof<CompoundKey, StateValue>),
    pub merge_group: (Option<Vec<(CompoundKey, StateValue)>>, MBTreeRangeProof<CompoundKey, StateValue>),
}

impl ColeStarInMemProof {
    pub fn new() -> Self {
        Self {
            write_group: (None, MBTreeRangeProof::default()),
            merge_group: (None, MBTreeRangeProof::default()),
        }
    }

    pub fn set_write_group(&mut self, r: Option<Vec<(CompoundKey, StateValue)>>, p: MBTreeRangeProof<CompoundKey, StateValue>) {
        self.write_group = (r, p);
    }

    pub fn set_merge_group(&mut self, r: Option<Vec<(CompoundKey, StateValue)>>, p: MBTreeRangeProof<CompoundKey, StateValue>) {
        self.merge_group = (r, p);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColeStarLevelProof {
    pub write_group: Vec<(Option<Vec<CompoundKeyValue>>, RunProofOrHash)>,
    pub merge_group: Vec<(Option<Vec<CompoundKeyValue>>, RunProofOrHash)>,
}

impl ColeStarLevelProof {
    pub fn new() -> Self {
        Self {
            write_group: Vec::new(),
            merge_group: Vec::new(),
        }
    }

    pub fn push_to_write_group(&mut self, r: Option<Vec<CompoundKeyValue>>, p: RunProofOrHash) {
        self.write_group.push((r, p));
    }

    pub fn push_to_merge_group(&mut self, r: Option<Vec<CompoundKeyValue>>, p: RunProofOrHash) {
        self.merge_group.push((r, p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use utils::{H160, H256, types::{AddrKey, CompoundKey}};
    #[test]
    fn test_insert_and_query() {
        let num_of_contract = 100;
        let num_of_addr = 100;
        let num_of_version = 500;
        let n = num_of_contract * num_of_addr * num_of_version;
        let mut rng = StdRng::seed_from_u64(1);

        let fanout = 15;
        let epsilon = 23;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 450000;
        let size_ratio = 10;
        let configs = Configs::new(fanout, epsilon, dir_name.to_string(), base_state_num, size_ratio);

        let mut state_vec = Vec::<CompoundKeyValue>::new();
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
                addr_key_vec.push(addr_key);
            }
        }
        // let multi_factor = 2;
        for k in 1..=num_of_version {
            for addr_key in &addr_key_vec {
                let compound_key = CompoundKey::new_with_addr_key(*addr_key, k * 2);
                state_vec.push(CompoundKeyValue::new_with_compound_key(compound_key, H256::from_low_u64_be((k * 2) as u64).into()));
            }
        }
/*         for addr_key in &addr_key_vec {
            for k in 1..=num_of_version {
                let compound_key = CompoundKey::new_with_addr_key(*addr_key, k * 2);
                state_vec.push(CompoundKeyValue::new_with_compound_key(compound_key, H256::from_low_u64_be((k * 2) as u64).into()));
            }
        } */

        let mut cole_star = ColeStar::new(&configs);
        let start = std::time::Instant::now();
        for state in &state_vec {
            cole_star.insert(*state);
        }
        let elapse = start.elapsed().as_nanos();
        println!("average insert: {:?}", elapse / n as u128);
        println!("cole_star memory cost: {:?}", cole_star.memory_cost());
        // println!("cole star: {:?}", cole_star);

        let start = std::time::Instant::now();
        for addr in &addr_key_vec {
            let r = cole_star.search_latest_state_value(*addr);
            let true_value = StateValue(H256::from_low_u64_be((num_of_version* 2) as u64));
            
            if r.unwrap() != true_value {
                println!("false addr: {:?}", addr);
                println!("r: {:?}, true value: {:?}", r, true_value);
                break;
            }
        }
        let elapse = start.elapsed().as_nanos();
        println!("average point query: {:?}", elapse / n as u128);
        let root = cole_star.compute_digest();
        let mut search_prove = 0;
        for addr in &addr_key_vec {
            let start = std::time::Instant::now();
            let proof = cole_star.search_with_proof(*addr, 0, num_of_version * 2);
            let (b, r) = verify_and_collect_result(*addr, 0, num_of_version * 2, root, &proof, configs.fanout);
            let elapse = start.elapsed().as_nanos();
            search_prove += elapse;
            let true_states: Vec<CompoundKeyValue> = (1..=num_of_version).map(|k| {
                let compound_key = CompoundKey::new_with_addr_key(*addr, k*2);
                CompoundKeyValue::new_with_compound_key(compound_key, H256::from_low_u64_be((k * 2) as u64).into())
            }).collect();
            if b == false {
                println!("false");
            }
            let r = r.unwrap();
            if true_states != r {
                println!("true states: {:?}", true_states);
                println!("r: {:?}", r);
            }
        }
        println!("avg search prove: {}", search_prove / (num_of_contract * num_of_addr) as u128);

        let mut search_prove = 0;
        for addr in &addr_key_vec {
            for i in 1..=num_of_version {
                let start = std::time::Instant::now();
                let proof = cole_star.search_with_proof(*addr, (2 * i) as u32, (2 * i) as u32);
                let (b, r) = verify_and_collect_result(*addr, (2 * i) as u32, (2 * i) as u32, root, &proof, configs.fanout);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                if b == false {
                    println!("false");
                }
                let r = r.unwrap();
                let compound_key = CompoundKey::new_with_addr_key(*addr, i * 2);
                let value = CompoundKeyValue::new_with_compound_key(compound_key, H256::from_low_u64_be((i * 2) as u64).into());
                let true_states = vec![value];
                if true_states != r {
                    println!("true states: {:?}", true_states);
                    println!("r: {:?}", r);
                }
            }
        }
        println!("avg search prove: {}", search_prove / (num_of_contract * num_of_addr * num_of_version) as u128);

        // std::thread::sleep(std::time::Duration::from_secs(30));
        // drop(cole_star);
        // let load = ColeStar::load(&configs);
        // println!("load: {:?}", load);
    }
}
