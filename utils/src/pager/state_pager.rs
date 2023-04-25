use crate::{File, OpenOptions, types::{CompoundKeyValue, CompoundKey, StateValue}, cacher::CacheManager};
use super::{Page, PAGE_SIZE, MAX_NUM_STATE_IN_PAGE};
use std::os::unix::prelude::FileExt;
/* A helper that writes the state into a file with a sequence of pages
   According to the disk-optimization objective, the state writing should be in a streaming fashion.
 */
pub struct StatePageWriter {
    pub file: File, // file object of the corresponding value file
    pub vec_in_latest_update_page: Vec<CompoundKeyValue>, // a preparation vector to obsorb the streaming state values which are not persisted in the file yet
    pub num_stored_pages: usize, // records the number of pages that are stored in the file
    pub num_states: usize, //records the number of states
}

impl StatePageWriter {
    /* Initialize the writer using a given file name
     */
    pub fn create(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(&file_name).unwrap();
        Self {
            file,
            vec_in_latest_update_page: vec![],
            num_stored_pages: 0,
            num_states: 0,
        }
    }

    /* Load the writer from a given file
    num_states and num_stored_pages are derived from the file
     */
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let mut num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut vec_in_latest_update_page = vec![];
        let mut num_states = 0;
        if num_stored_pages > 0 {
            let last_page_offset = (num_stored_pages - 1) * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            file.read_exact_at(&mut bytes, last_page_offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let page_vec = page.to_state_vec();
            // derive number of states in the last page
            let num_states_in_last_page = page_vec.len();
            // derive the number of states
            num_states = (num_stored_pages - 1) * MAX_NUM_STATE_IN_PAGE + num_states_in_last_page;

            if num_states_in_last_page != MAX_NUM_STATE_IN_PAGE {
                // the last page is not full, should not be finalized in the file
                vec_in_latest_update_page = page_vec;
                num_stored_pages -= 1;
            }
        }
        Self {
            file,
            vec_in_latest_update_page,
            num_stored_pages,
            num_states,
        }
    }

    /* Streamingly add the state to the latest_update_page
       Flush the latest_update_page to the file once it is full, and clear it.
     */
    pub fn append(&mut self, state: CompoundKeyValue) {
        // add the state
        self.vec_in_latest_update_page.push(state);
        if self.vec_in_latest_update_page.len() == MAX_NUM_STATE_IN_PAGE {
            // vector is full, should be added to a page and flushed the page to the file
            self.flush();
        }
        self.num_states += 1;
    }

    /* Flush the vector in latest update page to the last page in the value file
     */
    pub fn flush(&mut self) {
        if self.vec_in_latest_update_page.len() != 0 {
            // first put the vector into a page
            let page = Page::from_state_vec(&self.vec_in_latest_update_page);
            // compute the offset at which the page will be written in the file
            let offset = self.num_stored_pages * PAGE_SIZE;
            // write the page to the file
            self.file.write_all_at(&page.block, offset as u64).unwrap();
            // clear the vector
            self.vec_in_latest_update_page.clear();
            self.num_stored_pages += 1;
        }
    }

    /* Transform pager to reader
     */
    pub fn to_state_reader(self) -> StatePageReader {
        let num_states = self.num_states;
        let file = self.file;
        StatePageReader {
            file,
            num_states,
        }
    }

    /* Transform pager to iterator for preparing file merge
     */
    pub fn to_state_iter(self) -> StateIterator {
        let num_states = self.num_states;
        StateIterator {
            file: self.file,
            cur_vec_of_page: Vec::<CompoundKeyValue>::new(),
            cur_state_pos: 0,
            num_states,
        }
    }
}

/* A helper to read state from the file
   A LRU cache is used to optimize the read performance.
 */
pub struct StatePageReader {
    pub file: File, // file object of the corresponding value file
    pub num_states: usize, //records the number of states
}

impl StatePageReader {
    /* Load the reader from a given file. 
       num_states and num_stored_pages are derived from the file
     */
    pub fn load(file_name: &str) -> Self {
        let file = OpenOptions::new().create(true).read(true).write(true).open(&file_name).unwrap();
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut num_states = 0;
        if num_stored_pages > 0 {
            let last_page_offset = (num_stored_pages - 1) * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            file.read_exact_at(&mut bytes, last_page_offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let page_vec = page.to_state_vec();
            // derive number of states in the last page
            let num_states_in_last_page = page_vec.len();
            // derive the number of states
            num_states = (num_stored_pages - 1) * MAX_NUM_STATE_IN_PAGE + num_states_in_last_page;
        }
        Self {
            file,
            num_states,
        }
    }

    /* Load the deserialized vector of the page from the file at given page_id
     */
    pub fn read_deser_page_at(&mut self, run_id: u32, page_id: usize, cache_manager: &mut CacheManager) -> Vec<CompoundKeyValue> {
        // first check whether the cache contains the page
        let r = cache_manager.read_state_cache(run_id, page_id);
        if r.is_some() {
            // cache contains the page
            let v = r.unwrap().clone();
            return v;
        } else {
            // cache does not contain the page, should load the page from the file
            let offset = page_id * PAGE_SIZE;
            let mut v = [0u8; PAGE_SIZE];
            self.file.read_exact_at(&mut v, offset as u64).unwrap();
            let page = Page::from_array(v);
            let v = page.to_state_vec();
            // before return the vector, add it to the cache with page_id as the key
            cache_manager.set_state_cache(run_id, page_id, v.clone());
            return v;
        }
    }

    /* Load the deserialized vector given the state's location range
     */
    pub fn read_deser_states_range(&mut self, run_id: u32, pos_l: usize, pos_r: usize, cache_manager: &mut CacheManager) -> Vec<CompoundKeyValue> {
        let page_id_l = pos_l / MAX_NUM_STATE_IN_PAGE;
        let page_id_r = pos_r / MAX_NUM_STATE_IN_PAGE;
        let mut v = Vec::<CompoundKeyValue>::new();
        for page_id in page_id_l ..= page_id_r {
            let states = self.read_deser_page_at(run_id, page_id, cache_manager);
            v.extend(&states);
        }
        
        let left_slice_pos = pos_l % MAX_NUM_STATE_IN_PAGE;
        let right_slice_pos = (page_id_r - page_id_l) * MAX_NUM_STATE_IN_PAGE + (pos_r % MAX_NUM_STATE_IN_PAGE);
        v = v[left_slice_pos ..= right_slice_pos].to_vec();
        return v;
    }

    /* Transform the reader to iterator for preparing file merge, destroy the reader instance after the transformation.
     */
    pub fn to_state_iter(self) -> StateIterator {
        let num_states = self.num_states;
        StateIterator {
            file: self.file,
            cur_vec_of_page: Vec::<CompoundKeyValue>::new(),
            cur_state_pos: 0,
            num_states,
        }
    }
}

pub struct InMemStateIterator {
    pub states: Vec<(CompoundKey, StateValue)>,
    pub cur_state_pos: usize, // position of current state
}

impl InMemStateIterator {
    // create a new in-memory state iterator using the input state vector
    pub fn create(states: Vec<(CompoundKey, StateValue)>) -> Self {
        Self {
            states,
            cur_state_pos: 0,
        }
    }
}

/* Implementation of the iterator trait.
 */
impl Iterator for InMemStateIterator {
    // the data type of each iterated item is the state (compound key-value pair)
    type Item = (CompoundKey, StateValue);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_state_pos >= self.states.len() {
            // already reached the last state, return None 
            return None;
        } else {
            let r = self.states[self.cur_state_pos];
            self.cur_state_pos += 1;
            return Some(r);
        }
    }
}

/* Iterator of a state file
   Use a cached vector of page to fetch the state one-by-one in a streaming fashion.
   Note that the state should be read from the file in a 'Page' unit.
 */
pub struct StateIterator {
    pub file: File,
    pub cur_vec_of_page: Vec<CompoundKeyValue>, // cache of the current deserialized vector of page
    pub cur_state_pos: usize, // position of current state
    pub num_states: usize, // total number of states
}

impl StateIterator {
    /* Create a new state iterator by given the file handler and the number of states
     */
    pub fn create_with_num_states(file: File, num_states: usize) -> Self {
        Self {
            file,
            cur_vec_of_page: Vec::<CompoundKeyValue>::new(),
            cur_state_pos: 0,
            num_states,
        }
    }
    /* Create a new state iterator by given the file handler.
       The num_states is derived from the file (should load the last page to determine the number of states).
     */
    pub fn create(file: File) -> Self {
        let num_stored_pages = file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut num_states = 0;
        if num_stored_pages > 0 {
            let last_page_offset = (num_stored_pages - 1) * PAGE_SIZE;
            // get last page from file
            let mut bytes = [0u8; PAGE_SIZE];
            file.read_exact_at(&mut bytes, last_page_offset as u64).unwrap();
            let page = Page::from_array(bytes);
            let page_vec = page.to_state_vec();
            // derive number of states in the last page
            let num_states_in_last_page = page_vec.len();
            // derive the number of states
            num_states = (num_stored_pages - 1) * MAX_NUM_STATE_IN_PAGE + num_states_in_last_page;
        }
        Self {
            file,
            cur_vec_of_page: Vec::<CompoundKeyValue>::new(),
            cur_state_pos: 0,
            num_states,
        }
    }
}

/* Implementation of the iterator trait.
 */
impl Iterator for StateIterator {
    // the data type of each iterated item is the state (compound key-value pair)
    type Item = CompoundKeyValue;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_state_pos >= self.num_states {
            // already reached the last state, return None 
            return None;
        } else {
            // get the position inside the page
            let inner_page_pos = self.cur_state_pos % MAX_NUM_STATE_IN_PAGE;
            if inner_page_pos == 0 {
                // should fetch a new page from the file
                let mut bytes = [0u8; PAGE_SIZE];
                // get the page_id from the state position
                let page_id = self.cur_state_pos / MAX_NUM_STATE_IN_PAGE;
                let offset = page_id * PAGE_SIZE;
                self.file.read_exact_at(&mut bytes, offset as u64).unwrap();
                let page = Page::from_array(bytes);
                // deserialize the page to state vector
                self.cur_vec_of_page = page.to_state_vec();
            }
            // increment the state position
            self.cur_state_pos += 1;
            return Some(self.cur_vec_of_page[inner_page_pos]); // read the state from the vector
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{H160, H256};
    use caches::Cache;
    use rand::{rngs::StdRng, SeedableRng};
    use super::*;
    #[test]
    fn test_state_pager() {
        let n = 1000;
        let mut rng = StdRng::seed_from_u64(1);
        let mut writer = StatePageWriter::create("state.dat");
        let mut state_vec = Vec::<CompoundKeyValue>::new();
        for i in 0..n {
            let acc_addr = H160::random_using(&mut rng);
            let state_addr = H256::random_using(&mut rng);
            let version = i as u32;
            let value = H256::random_using(&mut rng);
            let state = CompoundKeyValue::new(acc_addr.into(), state_addr.into(), version, value.into());
            state_vec.push(state.clone());
            writer.append(state);
        }
        writer.flush();
        let mut reader = writer.to_state_reader();
        let mut cache_manager = CacheManager::new();
        for j in 0..5 {
            // iteratively read the pages
            let start = std::time::Instant::now();
            for i in 0..n {
                let page_id = i / MAX_NUM_STATE_IN_PAGE;
                let inner_page_pos = i % MAX_NUM_STATE_IN_PAGE;
                let v = reader.read_deser_page_at(0, page_id, &mut cache_manager);
                let state = v[inner_page_pos];
                assert_eq!(state, state_vec[i]);
            }
            let elapse = start.elapsed().as_nanos() as usize / n;
            println!("round {}, read state time: {}", j, elapse);
        }
        println!("size of cache: {}", cache_manager.state_cache.len());

        drop(reader);
        let pager = StatePageReader::load("state.dat");
        let mut it = pager.to_state_iter();
        let mut cnt = 0;
        loop {
            let r = it.next();
            if r.is_none() {
                break;
            } else {
                // let s = r.unwrap();
                // println!("{:?}", s);
                cnt += 1;
            }
        }
        assert_eq!(cnt, n);
    }
}