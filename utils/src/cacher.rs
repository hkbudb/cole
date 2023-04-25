use caches::{RawLRU, Cache};
use crate::{types::{CompoundKeyValue}, pager::{model_pager::ModelCollections, PAGE_SIZE}, H256};
pub const CACHE_SIZE: usize = 1024 * 1024; // 1MB
// each block is 4KB, therefore, the size of each cache is the capacity * 4KB

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageIndex {
    run_id: u32,
    page_id: usize,
}

impl PageIndex {
    pub fn new(run_id: u32, page_id: usize) -> Self {
        Self {
            run_id,
            page_id,
        }
    }
}

pub struct CacheManager {
    pub state_cache: RawLRU<PageIndex, Vec<CompoundKeyValue>>, 
    pub model_cache: RawLRU<PageIndex, ModelCollections>,
    pub mht_cache: RawLRU<PageIndex, Vec<H256>>
}

impl CacheManager {
    pub fn new() -> Self {
        let state_cache_capacity = ((CACHE_SIZE / PAGE_SIZE) as f64 * 0.4) as usize;
        let state_cache = RawLRU::<PageIndex, Vec<CompoundKeyValue>>::new(state_cache_capacity).unwrap();
        let model_cache_capacity = ((CACHE_SIZE / PAGE_SIZE) as f64 * 0.4) as usize;
        let model_cache = RawLRU::<PageIndex, ModelCollections>::new(model_cache_capacity).unwrap();
        let mht_cache_capacity = ((CACHE_SIZE / PAGE_SIZE) as f64 * 0.2) as usize;
        let mht_cache = RawLRU::<PageIndex, Vec<H256>>::new(mht_cache_capacity).unwrap();
        Self {
            state_cache,
            model_cache,
            mht_cache,
        }
    }

    pub fn read_state_cache(&mut self, run_id: u32, page_id: usize) -> Option<Vec<CompoundKeyValue>> {
        let index = PageIndex::new(run_id, page_id);
        self.state_cache.get(&index).cloned()
    }

    pub fn set_state_cache(&mut self, run_id: u32, page_id: usize, v: Vec<CompoundKeyValue>) {
        let index = PageIndex::new(run_id, page_id);
        self.state_cache.put(index, v);
    }

    pub fn read_model_cache(&mut self, run_id: u32, page_id: usize) -> Option<ModelCollections> {
        let index = PageIndex::new(run_id, page_id);
        self.model_cache.get(&index).cloned()
    }

    pub fn set_model_cache(&mut self, run_id: u32, page_id: usize, v: ModelCollections) {
        let index = PageIndex::new(run_id, page_id);
        self.model_cache.put(index, v);
    }

    pub fn read_mht_cache(&mut self, run_id: u32, page_id: usize) -> Option<Vec<H256>> {
        let index = PageIndex::new(run_id, page_id);
        self.mht_cache.get(&index).cloned()
    }

    pub fn set_mht_cache(&mut self, run_id: u32, page_id: usize, v: Vec<H256>) {
        let index = PageIndex::new(run_id, page_id);
        self.mht_cache.put(index, v);
    }

    pub fn compute_cacher_size(&self) -> (usize, usize, usize) {
        let state_cache_size = self.state_cache.keys().len() * (12 + PAGE_SIZE);
        let model_cache_size = self.model_cache.keys().len() * (12 + PAGE_SIZE);
        let mht_cache_size = self.mht_cache.keys().len() * (12 + PAGE_SIZE);
        (state_cache_size, model_cache_size, mht_cache_size)
    }
}