use serde::{Serialize, Deserialize};
use utils::{pager::{state_pager::{StatePageReader, StateIterator, InMemStateIterator}, model_pager::ModelPageReader, mht_pager::{HashPageReader, RangeProof, reconstruct_range}, PAGE_SIZE}, config::{compute_bitmap_size_in_bytes, Configs}, OpenOptions, Write, Read, merge_sort::{in_memory_merge, merge}, types::{CompoundKeyValue, CompoundKey, AddrKey, bytes_hash, Digestible}, models::CompoundKeyModel, cacher::CacheManager};
use utils::H256;
use std::cmp::{min, max};
use growable_bloom_filter::GrowableBloom;
use std::fmt::{Debug, Formatter, Error};
const FILTER_FP_RATE: f64 = 0.1; // false positive rate of a filter
const MAX_FILTER_SIZE: usize = 1024 * 1024; // 1M

// define a run in a level
pub struct LevelRun {
    pub run_id: u32, // id of this run
    pub state_reader: StatePageReader, // state's reader
    pub model_reader: ModelPageReader, // model's reader
    pub mht_reader: HashPageReader, // mht's reader
    pub filter: Option<GrowableBloom>, // bloom filter
    pub filter_hash: Option<H256>, // filter's hash
    pub digest: H256, // a digest of mht_root and filter's hash if filter exists
}

impl LevelRun {
    // load a run using the run's id, level's id and the configuration reference
    pub fn load(run_id: u32, level_id: u32, configs: &Configs) -> Self {
        // first define the file names of the state, model, mht, and filter
        let state_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "m");
        let mht_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "h");
        let filer_file_name = Self::file_name(run_id, level_id, &configs.dir_name, "f");

        // load the three readers using their file names
        let state_reader = StatePageReader::load(&state_file_name);
        let model_reader = ModelPageReader::load(&model_file_name);
        let mht_reader = HashPageReader::load(&mht_file_name);
        // initiate the filter using None object
        let mut filter = None;
        // if the filter's file exists, read the filter from the file
        match OpenOptions::new().read(true).open(&filer_file_name) {
            Ok(mut file) => {
                let mut len_bytes: [u8; 4] = [0x00; 4];
                file.read_exact(&mut len_bytes).unwrap();
                let len = u32::from_be_bytes(len_bytes);
                let mut v = vec![0u8; len as usize];
                file.read_exact(&mut v).unwrap();
                let filter_obj: GrowableBloom = bincode::deserialize(&v).unwrap();
                filter = Some(filter_obj);
            },
            Err(_) => {
            }
        }

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
        }
    }

    fn estimate_all_filter_size(level_id: u32, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> usize {
        let mut total_size = 0;
        let mut cur_level = level_id as i32;
        let mut cur_num_of_state = max_num_of_state;
        let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * level_num_of_run;
        total_size += cur_level_filter_size;
        while cur_level >= 0 {
            cur_level -= 1;
            cur_num_of_state /= size_ratio;
            let cur_level_filter_size = compute_bitmap_size_in_bytes(cur_num_of_state, FILTER_FP_RATE) * size_ratio;
            total_size += cur_level_filter_size;
        }
        return total_size;
    }
    // use the in-memory iterator to process the merge operation
    pub fn construct_run_by_in_memory_merge(inputs: Vec<InMemStateIterator>, run_id: u32, level_id: u32, dir_name: &str, epsilon: i64, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let mht_file_name = Self::file_name(run_id, level_id, &dir_name, "h");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_writer, model_writer, mht_writer, filter) = in_memory_merge(inputs, &state_file_name, &model_file_name, &mht_file_name, epsilon, fanout, filter);

        let state_reader = state_writer.to_state_reader();
        let model_reader = model_writer.to_model_reader();
        let mht_reader = mht_writer.to_hash_reader();

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
        }
    }
    // use the state iterator to process the merge operation
    pub fn construct_run_by_merge(inputs: Vec<StateIterator>, run_id: u32, level_id: u32, dir_name: &str, epsilon: i64, fanout: usize, max_num_of_state: usize, level_num_of_run: usize, size_ratio: usize) -> Self {
        let state_file_name = Self::file_name(run_id, level_id, &dir_name, "s");
        let model_file_name = Self::file_name(run_id, level_id, &dir_name, "m");
        let mht_file_name = Self::file_name(run_id, level_id, &dir_name, "h");
        // use level_id to determine whether we should create a filter for this run
        let est_filter_size = Self::estimate_all_filter_size(level_id, max_num_of_state, level_num_of_run, size_ratio);
        let mut filter = None;
        if est_filter_size <= MAX_FILTER_SIZE {
            filter = Some(GrowableBloom::new(FILTER_FP_RATE, max_num_of_state));
        }
        // merge the input states and construct the new state file, model file, mht file and insert state's keys to the filter
        let (state_writer, model_writer, mht_writer, filter) = merge(inputs, &state_file_name, &model_file_name, &mht_file_name, epsilon, fanout, filter);

        let state_reader = state_writer.to_state_reader();
        let model_reader = model_writer.to_model_reader();
        let mht_reader = mht_writer.to_hash_reader();

        let mht_root = mht_reader.root.unwrap();
        let mut filter_hash = None;
        if filter.is_some() {
            let filter_bytes = bincode::serialize(filter.as_ref().unwrap()).unwrap();
            filter_hash = Some(bytes_hash(&filter_bytes));
        }

        let digest = Self::load_digest(mht_root, &filter_hash);
        Self {
            run_id,
            state_reader,
            model_reader,
            mht_reader,
            filter,
            digest,
            filter_hash,
        }
    }
    
    // helper function to generate the file name of different file types: "s", "m", "h"
    pub fn file_name(run_id: u32, level_id: u32, dir_name: &str, file_type: &str) -> String {
        format!("{}/{}_{}_{}.dat", dir_name, file_type, level_id, run_id)
    }
    // persist the filter if it exists
    pub fn persist_filter(&self, level_id: u32, configs: &Configs) {
        if self.filter.is_some() {
            // init the filter's file name
            let filer_file_name = Self::file_name(self.run_id, level_id, &configs.dir_name, "f");
            // serialize the filter using bincode
            let bytes = bincode::serialize(self.filter.as_ref().unwrap()).unwrap();
            // get the length of the serialized bytes
            let bytes_len = bytes.len() as u32;
            // v is a vector that will be persisted to the filter's file
            let mut v = bytes_len.to_be_bytes().to_vec();
            v.extend(&bytes);
            // write v to the file
            let mut file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&filer_file_name).unwrap();
            file.write_all(&mut v).unwrap();
        }
    }

    pub fn search_run(&mut self, addr_key: AddrKey, configs: &Configs, cache_manager: &mut CacheManager) -> Option<CompoundKeyValue> {
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&addr_key) {
                return None;
            }
        }
        // use the model file to predict the position in the state file
        // compute the boundary compound key
        let upper_key = CompoundKey {
            addr: addr_key,
            version: u32::MAX,
        };
        let epsilon = configs.epsilon;
        // use model file to predict the pos
        let pred_pos = self.model_reader.get_pred_state_pos(self.run_id, &upper_key, epsilon, cache_manager) as i64;
        let num_of_states = self.state_reader.num_states;
        // compute the lower position and upper position according to the pred_pos and epsilon
        let pos_l = min(max(pred_pos - epsilon - 1, 0), num_of_states as i64 - 1) as usize;
        let pos_r = min(pred_pos + epsilon + 2, num_of_states as i64 - 1) as usize;
        // load the states from the value file given the range [pos_l, pos_r]
        let states = self.state_reader.read_deser_states_range(self.run_id, pos_l, pos_r, cache_manager);
        // binary search the loaded vector using the upper key
        let (_, res) = binary_search_of_key(&states, upper_key);
        if res.is_some() {
            let res = res.unwrap();
            let res_addr = res.key.addr;
            if res_addr == addr_key {
                return Some(res);
            }
        }
        return None;
    }

    fn prove_leaf(&mut self, l: usize, r: usize, num_of_data: usize, fanout: usize, proof: &mut RangeProof, cache_manager: &mut CacheManager) {
        let level_l = l;
        let level_r = r;
        let proof_pos_l = level_l - level_l % fanout;
        let proof_pos_r = if level_r - level_r % fanout + fanout > num_of_data {
            num_of_data
        } else {
            level_r - level_r % fanout + fanout
        } - 1;
        let states = self.state_reader.read_deser_states_range(self.run_id, proof_pos_l, proof_pos_r, cache_manager);
        let mut leaf_hashes: Vec<H256> = states.iter().map(|s| s.to_digest()).collect();
        for _ in 0..(level_r - level_l + 1) {
            leaf_hashes.remove(level_l - proof_pos_l);
        }

        proof.p.insert(0, leaf_hashes);
    }

    /* Generate the result and the RunProof
    If the filter does show that the addr_key does not exist, use the filter + MHT root as the proof.
    If the filter cannot show, use the MHT to prove the result, add the filter's hash to the proof.
     */
    pub fn prove_range(&mut self, addr_key: AddrKey, low_version: u32, upper_version: u32, configs: &Configs, cache_manager: &mut CacheManager) -> (Option<Vec<CompoundKeyValue>>, RunProof) {
        // init the proof
        let mut proof = RunProof::new();
        // try to use filter to test whether the key exists
        if self.filter.is_some() {
            if !self.filter.as_ref().unwrap().contains(&addr_key) {
                // the addr_key must not exist in the run, so include the filter to the proof
                let filter = self.filter.clone().unwrap();
                let mht_root = self.mht_reader.root.unwrap();
                proof.include_filter_and_mht_root(filter, mht_root);
                return (None, proof);
            }
        }

        // compute the boundary compound key
        let lower_key = CompoundKey::new_with_addr_key(addr_key, low_version);
        let upper_key = CompoundKey::new_with_addr_key(addr_key, upper_version);
        let epsilon = configs.epsilon;
        // use model file to predict the pos
        let pred_pos_low = self.model_reader.get_pred_state_pos(self.run_id, &lower_key, epsilon, cache_manager) as i64;
        let pred_pos_upper = self.model_reader.get_pred_state_pos(self.run_id, &upper_key, epsilon, cache_manager) as i64;
        let num_of_states = self.state_reader.num_states;
        // compute the lower position and upper position according to the pred_pos and epsilon
        let pos_l = min(max(pred_pos_low - epsilon - 1, 0), num_of_states as i64 - 1) as usize;
        let pos_r = min(pred_pos_upper + epsilon + 3, num_of_states as i64 - 1) as usize;
        // load the states from the value file given the range [pos_l, pos_r]
        let states = self.state_reader.read_deser_states_range(self.run_id, pos_l, pos_r, cache_manager);
        // binary search the keys in the retrieved vector
        let (upper_index_inner, _) = binary_search_of_key(&states, upper_key);
        let (lower_index_inner, _) = binary_search_of_key(&states, lower_key);
        // derive the actual position by adding offset pos_l
        let mut left_proof_pos = lower_index_inner + pos_l;
        let mut right_proof_pos = upper_index_inner + pos_l;
        // two boundary positions
        if left_proof_pos != 0 && states[lower_index_inner].key == lower_key {
            // not the first element, require a less-one boundary
            left_proof_pos -= 1;
        }
        if right_proof_pos != num_of_states - 1 {
            // right_proof_pos is not the right-most pos
            right_proof_pos += 1;
        }
        // result key_value pairs
        let data_vec = states[left_proof_pos - pos_l ..= right_proof_pos - pos_l].to_vec();
        // generate non_leaf range_proof
        let fanout = configs.fanout;
        let mut range_proof = self.mht_reader.prove_non_leaf(self.run_id, left_proof_pos, right_proof_pos, num_of_states, fanout, cache_manager);
        // generate leaf range_proof
        self.prove_leaf(left_proof_pos, right_proof_pos, num_of_states, fanout, &mut range_proof, cache_manager);
        
        // compute the filter's hash and add it to the Run's proof
        let mut filter_hash = None;
        if self.filter_hash.is_some() {
            filter_hash = Some(self.filter_hash.unwrap());
        }
        proof.include_range_proof_and_filer_hash(range_proof, filter_hash);
        return (Some(data_vec), proof);
    }

    // derive the digest of the LevelRun
    pub fn compute_digest(&self) -> H256 {
        let mht_root = self.mht_reader.root.unwrap();
        let mut bytes = mht_root.as_bytes().to_vec();
        if self.filter_hash.is_some() {
            bytes.extend(self.filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    // compute the digest of the run according to the MHT root and the filter if it exists
    pub fn load_digest(mht_root: H256, filter_hash: &Option<H256>) -> H256 {
        let mut bytes = mht_root.as_bytes().to_vec();
        if filter_hash.is_some() {
            bytes.extend(filter_hash.unwrap().as_bytes());
        }
        bytes_hash(&bytes)
    }

    pub fn format_run(&mut self, cache_manager: &mut CacheManager) {
        println!("run id: {}", self.run_id);
        println!("filter: {:?}", self.filter);
        let state_page_num = self.state_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut states = Vec::<CompoundKeyValue>::new();
        for page_id in 0..state_page_num {
            let v = self.state_reader.read_deser_page_at(self.run_id, page_id, cache_manager);
            states.extend(&v);
        }
        // println!("{:?}", states);
        println!("states len: {}", states.len());
        let mut models = Vec::<CompoundKeyModel>::new();
        let model_page_num = self.model_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..model_page_num {
            let v = self.model_reader.read_deser_page_at(self.run_id, page_id, cache_manager).v;
            models.extend(&v);
        }
        println!("model len: {}", models.len());

        let mut hashes = Vec::<H256>::new();
        let hash_page_num = self.mht_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..hash_page_num {
            let v = self.mht_reader.read_deser_page_at(self.run_id, page_id, cache_manager);
            hashes.extend(&v);
        }
        println!("hash len: {}", hashes.len());
        println!("root: {:?}", hashes.last().unwrap());
        println!("cached root: {:?}", self.mht_reader.root);
    }

    pub fn load_states(&mut self, cache_manager: &mut CacheManager) -> Vec<CompoundKeyValue> {
        let state_page_num = self.state_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut states = Vec::<CompoundKeyValue>::new();
        for page_id in 0..state_page_num {
            let v = self.state_reader.read_deser_page_at(self.run_id, page_id, cache_manager);
            states.extend(&v);
        }
        return states;
    }

    pub fn print_models(&mut self, cache_manager: &mut CacheManager) {
        let mut models = Vec::<CompoundKeyModel>::new();
        let model_page_num = self.model_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..model_page_num {
            let v = self.model_reader.read_deser_page_at(self.run_id, page_id, cache_manager).v;
            models.extend(&v);
        }
        println!("model: {:?}", models);
    }

    pub fn filter_cost(&self) -> RunFilterSize {
        // filter cost
        let filter_ref = self.filter.as_ref();
        let filter_size;
        if filter_ref.is_some() {
            filter_size = filter_ref.unwrap().memory_size();
        } else {
            filter_size = 0;
        }
        return RunFilterSize::new(filter_size);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RunFilterSize {
    pub filter_size: usize,
}

impl RunFilterSize {
    pub fn new(filter_size: usize) -> Self {
        Self {
            filter_size,
        }
    }

    pub fn add(&mut self, other: &RunFilterSize) {
        self.filter_size += other.filter_size;
    }
}

impl Debug for LevelRun {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
/*         let mut_ref = unsafe {
            let const_ptr = self as *const LevelRun;
            let mut_ptr  = const_ptr as *mut LevelRun;
            &mut *mut_ptr 
        };
        let load_states = mut_ref.load_states();
        println!("run states: {:?}", load_states); */
        write!(f, "<Level Run Info> run_id: {}, filter is some: {:?}, states len: {}, cached mht root: {:?}, digest: {:?}", self.run_id, self.filter.is_some(), self.state_reader.num_states, self.mht_reader.root, self.digest)
    }
}

/* Either a filter or the digest of the filter
   used to be the part of the proof to prove the non-existence of the addr
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOrHash {
    Filter(GrowableBloom),
    Hash(H256),
}

/* Either a range proof or the MHT root hash
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeProofOrHash {
    RangeProof(RangeProof),
    Hash(H256),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunProof {
    range_proof_or_hash: RangeProofOrHash,
    filter_or_hash: Option<FilterOrHash>,
}

impl RunProof {
    // init the run proof with the empty range proof and empty filter or hash
    pub fn new() -> Self {
        let range_proof_or_hash = RangeProofOrHash::Hash(H256::default());
        let filter_or_hash = None;
        Self {
            range_proof_or_hash,
            filter_or_hash,
        }
    }

    pub fn include_filter_and_mht_root(&mut self, filter: GrowableBloom, root: H256) {
        let range_proof_or_hash = RangeProofOrHash::Hash(root);
        let filter_or_hash = Some(FilterOrHash::Filter(filter));
        self.filter_or_hash = filter_or_hash;
        self.range_proof_or_hash = range_proof_or_hash;
    }

    pub fn include_range_proof_and_filer_hash(&mut self, range_proof: RangeProof, filter_hash: Option<H256>) {
        let range_proof_or_hash = RangeProofOrHash::RangeProof(range_proof);
        let mut filter_or_hash = None;
        if filter_hash.is_some() {
            filter_or_hash = Some(FilterOrHash::Hash(filter_hash.unwrap()));
        }
        self.filter_or_hash = filter_or_hash;
        self.range_proof_or_hash = range_proof_or_hash;
    }

    pub fn get_filter_ref(&self) -> Option<&GrowableBloom> {
        match &self.filter_or_hash {
            Some(filter_or_hash) => {
                match filter_or_hash {
                    FilterOrHash::Filter(f) => Some(f),
                    FilterOrHash::Hash(_) => None,
                }
            },
            None => return None,
        }
    }

    pub fn get_range_proof_ref(&self) -> Option<&RangeProof> {
        match &self.range_proof_or_hash {
            RangeProofOrHash::RangeProof(p) => Some(p),
            RangeProofOrHash::Hash(_) => None,
        }
    }

    pub fn get_filter_hash(&self) -> Option<H256> {
        match &self.filter_or_hash {
            Some(filter_or_hash) => {
                match filter_or_hash {
                    FilterOrHash::Filter(_) => None,
                    FilterOrHash::Hash(h) => Some(*h),
                }
            },
            None => return None,
        }
    }

    pub fn get_merkle_hash(&self) -> Option<H256> {
        match &self.range_proof_or_hash {
            RangeProofOrHash::Hash(h) => Some(*h),
            RangeProofOrHash::RangeProof(_) => None,
        }
    }
}

pub fn reconstruct_run_proof(addr_key: AddrKey, low_version: u32, upper_version: u32, results: &Option<Vec<CompoundKeyValue>>, proof: &RunProof, fanout: usize) -> (bool, H256) {
    if results.is_none() {
        // there is no result, the filter must filter the addr_key and the range proof is empty
        let filter_ref = proof.get_filter_ref();
        let merkle_root = proof.get_merkle_hash();
        if filter_ref.is_none() || merkle_root.is_none() {
            return (false, H256::default());
        } else {
            let filter = filter_ref.unwrap();
            let root = merkle_root.unwrap();
            if filter.contains(addr_key) {
                // the filter contains the addr_key, but it should not, so return false
                return (false, H256::default());
            } else {
                // filter does not contain addr_key, it successfully prove the non-existence, combine the filter's hash and merkle root hash
                let filter_bytes = bincode::serialize(filter).unwrap();
                let filter_hash = bytes_hash(&filter_bytes);
                let mut bytes = root.as_bytes().to_vec();
                bytes.extend(filter_hash.as_bytes());
                let recomputed_h = bytes_hash(&bytes);
                return (true, recomputed_h);
            }
        }
    } else {
        // there is result, the filter must not prove the non-existence and the range proof is not empty
        let range_proof_ref = proof.get_range_proof_ref();
        if range_proof_ref.is_none() {
            return (false, H256::default());
        }
        let filter_hash = proof.get_filter_hash();
        let range_proof = range_proof_ref.unwrap();
        let objs = results.as_ref().unwrap();
        let obj_hashes: Vec<H256> = objs.iter().map(|elem| elem.to_digest()).collect();
        let reconstruct_merkle_root = reconstruct_range(range_proof, fanout, obj_hashes);
        let mut bytes = reconstruct_merkle_root.as_bytes().to_vec();
        if filter_hash.is_some() {
            bytes.extend(filter_hash.unwrap().as_bytes());
        }
        let recomputed_h = bytes_hash(&bytes);
        let lower_key = CompoundKey::new_with_addr_key(addr_key, low_version);
        let upper_key = CompoundKey::new_with_addr_key(addr_key, upper_version);
        for (i, data) in objs.iter().enumerate() {
            let data_key = data.key;
/*             if i != 0 && i != objs.len() - 1 {
                if data_key < lower_key || data_key > upper_key {
                    return (false, H256::default());
                }
            } */
            if i == 0 {
                if data_key >= lower_key {
                    println!("first");
                    return (false, H256::default());
                }
            } else if i == objs.len() - 1 {
                if data_key <= upper_key{
                    println!("last");
                    return (false, H256::default());
                }
            } else {
                if data_key < lower_key || data_key > upper_key {
                    return (false, H256::default());
                }
            }
        }
        return (true, recomputed_h);
    }
}
pub  fn verify_run_proof(addr_key: AddrKey, low_version: u32, upper_version: u32, results: &Option<Vec<CompoundKeyValue>>, proof: &RunProof, fanout: usize, root_hash: H256) -> bool {
    if results.is_none() {
        // there is no result, the filter must filter the addr_key and the range proof is empty
        let filter_ref = proof.get_filter_ref();
        let merkle_root = proof.get_merkle_hash();
        if filter_ref.is_none() || merkle_root.is_none() {
            return false;
        } else {
            let filter = filter_ref.unwrap();
            let root = merkle_root.unwrap();
            if filter.contains(addr_key) {
                // the filter contains the addr_key, but it should not, so return false
                return false;
            } else {
                // filter does not contain addr_key, it successfully prove the non-existence, combine the filter's hash and merkle root hash
                let filter_bytes = bincode::serialize(filter).unwrap();
                let filter_hash = bytes_hash(&filter_bytes);
                let mut bytes = root.as_bytes().to_vec();
                bytes.extend(filter_hash.as_bytes());
                let recomputed_h = bytes_hash(&bytes);
                if recomputed_h != root_hash {
                    // the run's digest mismatches
                    return false;
                } else {
                    return true;
                }
            }
        }
    } else {
        // there is result, the filter must not prove the non-existence and the range proof is not empty
        let range_proof_ref = proof.get_range_proof_ref();
        if range_proof_ref.is_none() {
            return false;
        }
        let filter_hash = proof.get_filter_hash();
        let range_proof = range_proof_ref.unwrap();
        let objs = results.as_ref().unwrap();
        let obj_hashes: Vec<H256> = objs.iter().map(|elem| elem.to_digest()).collect();
        let reconstruct_merkle_root = reconstruct_range(range_proof, fanout, obj_hashes);
        let mut bytes = reconstruct_merkle_root.as_bytes().to_vec();
        if filter_hash.is_some() {
            bytes.extend(filter_hash.unwrap().as_bytes());
        }
        let recomputed_h = bytes_hash(&bytes);
        if recomputed_h != root_hash {
            // the run's digest mismatches
            return false;
        } else {
            let lower_key = CompoundKey::new_with_addr_key(addr_key, low_version);
            let upper_key = CompoundKey::new_with_addr_key(addr_key, upper_version);
            for (i, data) in objs.iter().enumerate() {
                let data_key = data.key;
                if i == 0 {
                    if data_key >= lower_key {
                        println!("first");
                        return false;
                    }
                } else if i == objs.len() - 1 {
                    if data_key <= upper_key{
                        println!("last");
                        return false;
                    }
                } else {
                    if data_key < lower_key || data_key > upper_key {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}

pub fn binary_search_of_key(v: &Vec<CompoundKeyValue>, key: CompoundKey) -> (usize, Option<CompoundKeyValue>) {
    let mut index: usize;
    let len = v.len();
    let mut l: i32 = 0;
    let mut r: i32 = len as i32 - 1;
    if len == 0 {
        return (0, None);
    }

    while l <= r && l >=0 && r <= len as i32 - 1{
        let m = l + (r - l) / 2;
        if v[m as usize].key < key {
            l = m + 1;
        }
        else if v[m as usize].key > key {
            r = m - 1;
        }
        else {
            index = m as usize;
            return (index, Some(v[index].clone()));
        }
    }
    
    index = l as usize;
    if index == len {
        index -= 1;
    }

    if key < v[index].key && index > 0 {
        index -= 1;
    }
    return (index, Some(v[index].clone()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use utils::pager::{state_pager::StatePageWriter};
    use utils::types::{CompoundKeyValue, CompoundKey, StateValue};
    use utils::{H160, H256};
    use utils::config::Configs;
    #[test]
    fn generate_states() {
        let k: usize = 2;
        let n: usize = 10;
        let mut rng = StdRng::seed_from_u64(1);
        let mut pagers = Vec::<StatePageWriter>::new();
        for i in 0..k {
            let mut pager = StatePageWriter::create(&format!("data{}.dat", i));
            let mut state_vec = Vec::<CompoundKeyValue>::new();
            for i in 0..n {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let version = i as u32;
                let value = H256::random_using(&mut rng);
                let state = CompoundKeyValue::new(acc_addr.into(), state_addr.into(), version, value.into());
                state_vec.push(state.clone());
            }
            let min_key = CompoundKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into(), 0);
            let max_key = CompoundKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX);
            state_vec.push(CompoundKeyValue::new_with_compound_key(min_key, H256::default().into()));
            state_vec.push(CompoundKeyValue::new_with_compound_key(max_key, H256::default().into()));
            state_vec.sort();
            for s in state_vec {
                pager.append(s);
            }
            pager.flush();
            pagers.push(pager);
        }
    }

    #[test]
    fn test_hash() {
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }

        std::fs::create_dir(dir_name).unwrap_or_default();
        let mut rng = StdRng::seed_from_u64(1);
        let acc_addr = H160::random_using(&mut rng);
        let state_addr = H256::random_using(&mut rng);
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
        let min_key = CompoundKey::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into(), 0);
        let max_key = CompoundKey::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX);
        state_vec.push((min_key, StateValue(H256::default())));
        state_vec.push((CompoundKey::new(acc_addr.into(), state_addr.into(), 34), StateValue(H256::from_low_u64_be(34))));
        state_vec.push((CompoundKey::new(acc_addr.into(), state_addr.into(), 36), StateValue(H256::from_low_u64_be(36))));
        state_vec.push((max_key, StateValue(H256::default())));
        let it = InMemStateIterator::create(state_vec);
        let iters = vec![it];
        let epsilon = 23;
        let fanout = 5;
        let n = 2;
        let run_id = 1;
        let level_id = 0;
        let k = 2;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n as usize,
            size_ratio: k as usize,
        };

        let mut run = LevelRun::construct_run_by_in_memory_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, k as usize);
        println!("run: {:?}", run);
        let total_root_hash = run.digest;

        let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
        let lb = 35;
        let ub = 37;
        let mut cache_manager = CacheManager::new();
        let (r, p) = run.prove_range(addr_key, lb, ub, &configs, &mut cache_manager);
        println!("{:?}", r);
        let (b, re_h) = reconstruct_run_proof(addr_key, lb, ub, &r, &p, configs.fanout);
        if b == false || re_h != total_root_hash {
            println!("false");
        }

        let random_key = AddrKey::new(H160::random().into(), H256::random().into());
        let (r, p) = run.prove_range(random_key, lb, ub, &configs, &mut cache_manager);
        let (b, re_h) = reconstruct_run_proof(random_key, lb, ub, &r, &p, configs.fanout);
        if b == false || re_h != total_root_hash {
            println!("false");
        }
    }
    #[test]
    fn test_in_memory_merge_and_run_construction() {
        let k: usize = 2;
        let n: usize = 10;
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 46;
        let fanout = 2;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }

        std::fs::create_dir(dir_name).unwrap_or_default();
        let mut iters = Vec::new();
        for _ in 0..k {
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
        }

        let run_id = 1;
        let level_id = 0;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n as usize,
            size_ratio: k as usize,
        };

        let run = LevelRun::construct_run_by_in_memory_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, k as usize);
        run.persist_filter(level_id, &configs);
        drop(run);
        let load_run = LevelRun::load(run_id, level_id, &configs);
        println!("{:?}", load_run);

        let mht_root = load_run.mht_reader.root.unwrap();
        let mut bytes = mht_root.as_bytes().to_vec();
        let filter = load_run.filter.as_ref().unwrap();
        let filter_bytes = bincode::serialize(filter).unwrap();
        let filter_digest = bytes_hash(&filter_bytes);
        bytes.extend(filter_digest.as_bytes());
        let computed_h = bytes_hash(&bytes);
        assert_eq!(computed_h, load_run.digest);
    }
    #[test]
    fn test_run_construct_and_load() {
        let k = 2;
        let n = 10;
        let epsilon = 46;
        let fanout = 2;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }

        std::fs::create_dir(dir_name).unwrap_or_default();
        let mut iters = Vec::<StateIterator>::new();
        for i in 0..k {
            let file = OpenOptions::new().create(true).read(true).write(true).open(&format!("data{}.dat", i)).unwrap();
            let iter = StateIterator::create(file);
            iters.push(iter);
        }

        let run_id = 1;
        let level_id = 0;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n as usize,
            size_ratio: k as usize,
        };

        let run = LevelRun::construct_run_by_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, k as usize);
        run.persist_filter(level_id, &configs);
        drop(run);
        let mut load_run = LevelRun::load(run_id, level_id, &configs);
        let mut cache_manager = CacheManager::new();
        load_run.format_run(&mut cache_manager);
    }

    #[test]
    fn test_prove() {
        let num_of_contract = 34;
        let num_of_addr = 56;
        let num_of_version = 12;
        let mut rng = StdRng::seed_from_u64(1);
        let epsilon = 46;
        let fanout = 5;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }

        std::fs::create_dir(dir_name).unwrap_or_default();
        let mut iters = Vec::new();
        let mut state_vec = Vec::<(CompoundKey, StateValue)>::new();
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
        for addr_key in &addr_key_vec {
            for k in 1..=num_of_version {
                let compound_key = CompoundKey::new_with_addr_key(*addr_key, k * 2);
                let value = H256::from_low_u64_be((k * 2) as u64);
                state_vec.push((compound_key, StateValue(value)));
            }
        }
        state_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let it = InMemStateIterator::create(state_vec);
        iters.push(it);

        let n = num_of_contract * num_of_addr * num_of_version;
        let run_id = 1;
        let level_id = 0;
        let configs = Configs {
            fanout,
            epsilon,
            dir_name: dir_name.to_string(),
            base_state_num: n as usize,
            size_ratio: 1,
        };

        let run = LevelRun::construct_run_by_in_memory_merge(iters, run_id, level_id, &configs.dir_name, configs.epsilon, configs.fanout, configs.max_num_of_states_in_a_run(level_id), 1, 1);
        run.persist_filter(level_id, &configs);
        drop(run);
        let mut load_run = LevelRun::load(run_id, level_id, &configs);
        let mut cache_manager = CacheManager::new();
        println!("run: {:?}", load_run);
        let total_root_hash = load_run.digest;
        for addr_key in &addr_key_vec {
            for k in 1..=num_of_version {
                let lb = 2 * k;
                let ub = 2 * k;
                let (r, p) = load_run.prove_range(*addr_key, lb, ub, &configs, &mut cache_manager);
                let (b, re_h) = reconstruct_run_proof(*addr_key, lb, ub, &r, &p, configs.fanout);
                if b == false || re_h != total_root_hash {
                    println!("false");
                }
            }
        }

        let random_key = AddrKey::new(H160::random().into(), H256::random().into());
        let lb = 3;
        let ub = 4;
        let (r, p) = load_run.prove_range(random_key, lb, ub, &configs, &mut cache_manager);
        let (b, re_h) = reconstruct_run_proof(random_key, lb, ub, &r, &p, configs.fanout);
        if b == false || re_h != total_root_hash {
            println!("false");
        }
    }
}
