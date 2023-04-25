pub mod state_pager;
pub mod model_pager;
pub mod mht_pager;
use model_pager::ModelCollections;
use super::{types::{CompoundKeyValue, STATE_SIZE}, models::{CompoundKeyModel, MODEL_SIZE}};
use crate::H256;
pub const PAGE_SIZE: usize = 4096;
pub const MAX_NUM_STATE_IN_PAGE: usize = PAGE_SIZE / STATE_SIZE;
pub const MAX_NUM_MODEL_IN_PAGE: usize = PAGE_SIZE / MODEL_SIZE;
pub const MAX_NUM_HASH_IN_PAGE: usize = PAGE_SIZE / 32 - 1; // dedeuction of one is because we need to store some meta-data (i.e., num_of_hash) in the page

/* Structure of the page with default 4096 byte
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Page {
    pub block: [u8; PAGE_SIZE], // 4096 byte page size
}

impl Page {
    /* Initialize a page with 4096 bytes
     */
    pub fn new() -> Self {
        Self {
            block: [0u8; PAGE_SIZE],
        }
    }

    /* Create a page with the given data block
     */
    pub fn from_array(block: [u8; PAGE_SIZE]) -> Self {
        Self {
            block,
        }
    }

    /* Write hash vector to the page
    4 bytes num of hash | hash_0, hash_1, ...
     */
    pub fn from_hash_vec(v: &Vec<H256>) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_HASH_IN_PAGE);
        let mut page = Self::new();
        let num_of_hash = v.len() as u32;
        let num_of_hash_bytes = num_of_hash.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of hash in the front of the page
        offset.copy_from_slice(&num_of_hash_bytes);
        for (i, hash) in v.iter().enumerate() {
            let bytes = hash.as_bytes();
            let start_idx = i * 32 + 4;
            let end_idx = start_idx + 32;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(bytes);
        }
        return page;
    }

    /*
    Read the hashes from a block
     */
    pub fn to_hash_vec(&self) -> Vec<H256> {
        let mut v = Vec::<H256>::new();
        // deserialize the number of hashes in the page
        let num_of_hash = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize each of the hash from the page
        for i in 0..num_of_hash {
            let start_idx = i * 32 + 4;
            let end_idx = start_idx + 32;
            let hash = H256::from_slice(&self.block[start_idx .. end_idx]);
            v.push(hash);
        }
        return v;
    }

    /* Write model vector to the page
    4 bytes num of model | 4 bytes model level | model_0, model_1, ...
     */
    pub fn from_model_vec(v: &Vec<CompoundKeyModel>, model_level: u32) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_MODEL_IN_PAGE);
        let mut page = Self::new();
        let num_of_model = v.len() as u32;
        let num_of_model_bytes = num_of_model.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of models in the front of the page
        offset.copy_from_slice(&num_of_model_bytes);
        let model_level_bytes = model_level.to_be_bytes();
        let offset = &mut page.block[4..8];
        // write model level
        offset.copy_from_slice(&model_level_bytes);
        // iteratively write each model to the block
        for (i, model) in v.iter().enumerate() {
            let bytes = model.to_bytes();
            let start_idx = i * MODEL_SIZE + 8;
            let end_idx = start_idx + MODEL_SIZE;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(&bytes);
        }
        return page;
    }

    /*
    Read the models from a block
     */
    pub fn to_model_vec(&self) -> ModelCollections {
        let mut v = Vec::<CompoundKeyModel>::new();
        // deserialize the number of models in the page
        let num_of_model = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize the model level
        let model_level = u32::from_be_bytes(self.block[4..8].try_into().expect("error"));
        // deserialize each of the model from the page
        for i in 0..num_of_model {
            let start_idx = i * MODEL_SIZE + 8;
            let end_idx = start_idx + MODEL_SIZE;
            let model = CompoundKeyModel::from_bytes(&self.block[start_idx .. end_idx]);
            v.push(model);
        }
        return ModelCollections {
            v,
            model_level,
        };
    }

    /*
    Write state vector to the page
    4 bytes num of state | state_0, state_1, ...
     */
    pub fn from_state_vec(v: &Vec<CompoundKeyValue>) -> Self {
        // check the length of the vector is inside the maximum number in page
        assert!(v.len() <= MAX_NUM_STATE_IN_PAGE);
        let mut page = Self::new();
        let num_of_state = v.len() as u32;
        let num_of_state_bytes = num_of_state.to_be_bytes();
        let offset = &mut page.block[0..4];
        // write the number of states in the front of the page
        offset.copy_from_slice(&num_of_state_bytes);
        // iteratively write each state to the block
        for (i, state) in v.iter().enumerate() {
            let bytes = state.to_bytes();
            let start_idx = i * STATE_SIZE + 4;
            let end_idx = start_idx + STATE_SIZE;
            let offset = &mut page.block[start_idx .. end_idx];
            offset.copy_from_slice(&bytes);
        }
        return page;
    }
    /*
    Read the states from a block
     */
    pub fn to_state_vec(&self) -> Vec<CompoundKeyValue> {
        let mut v = Vec::<CompoundKeyValue>::new();
        // deserialize the number of states in the page
        let num_of_state = u32::from_be_bytes(self.block[0..4].try_into().expect("error")) as usize;
        // deserialize each of the state from the page
        for i in 0..num_of_state {
            let start_idx = i * STATE_SIZE + 4;
            let end_idx = start_idx + STATE_SIZE;
            let state = CompoundKeyValue::from_bytes(&self.block[start_idx .. end_idx]);
            v.push(state);
        }
        return v;
    }
}

#[cfg(test)]
mod tests {
    use crate::{H160, H256};
    use rand::{rngs::StdRng, SeedableRng};
    use super::*;
    #[test]
    fn test_state_and_page_ser_deser() {
        let mut rng = StdRng::seed_from_u64(1);
        for num_in_page in 0..MAX_NUM_STATE_IN_PAGE {
            let mut v = Vec::<CompoundKeyValue>::new();
            for i in 0..num_in_page {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let version = i as u32;
                let value = H256::random_using(&mut rng);
                let state = CompoundKeyValue::new(acc_addr.into(), state_addr.into(), version, value.into());
                v.push(state);
            }
            let page = Page::from_state_vec(&v);
            let deser_v = page.to_state_vec();
            assert_eq!(deser_v, v);
        }
    }
}