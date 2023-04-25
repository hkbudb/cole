pub mod mbtree;
use std::collections::{BTreeMap};
use merkle_btree_storage::{traits::BPlusTreeNodeIO, RangeProof as MBTreeProof, WriteContext};
use non_persist_trie::{NonPersistMPT, Proof as MPTProof, PointerHash, verify_with_addr_key, Value};
use utils::{types::{AddrKey, StateValue}, H256, ROCKSDB_COL_ID};
use serde::{Serialize, Deserialize};
use mbtree::{PersistMBTree, VersionKey};
use kvdb_rocksdb::Database;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct TotalProof {
    pub upper_proof: (Option<(u64, H256)>, MPTProof), // Some(key_id, low_root_h), mpt_proof
    pub lower_proof: Option<MBTreeProof<VersionKey, StateValue>>,
}

pub struct NonLearnCMI<'a> {
    pub key_counter: u64, // use to generate the lower mb-tree's tree_id
    pub upper_tree: NonPersistMPT<'a>,
    pub mb_tree_fanout: usize,
    pub db: &'a Database,
}

impl<'a> Drop for NonLearnCMI<'a> {
    fn drop(&mut self) {
        let key_counter_load_key = "key_counter".as_bytes().to_vec();
        let key_counter_bytes = self.key_counter.to_be_bytes().to_vec();
        let mut tx = self.db.transaction();
        tx.put(ROCKSDB_COL_ID, &key_counter_load_key, &key_counter_bytes);
        self.db.write(tx).unwrap();
    }
}

impl<'a> NonLearnCMI<'a> {
    pub fn new(db: &'a Database, mb_tree_fanout: usize) -> Self {
        let upper_tree = NonPersistMPT::new(db);
        Self {
            key_counter: 0,
            upper_tree,
            mb_tree_fanout,
            db,
        }
    }

    pub fn open(db: &'a Database, mb_tree_fanout: usize) -> Self {
        let upper_tree = NonPersistMPT::open(&db);
        // load key_counter
        let key_counter_load_key = "key_counter".as_bytes().to_vec();
        let key_counter_bytes = db.get(ROCKSDB_COL_ID, &key_counter_load_key).unwrap();
        let mut key_counter = 0;
        match key_counter_bytes {
            Some(v) => {
                key_counter = u64::from_be_bytes(v.try_into().unwrap());
            },
            None => {

            }
        }
        Self {
            key_counter,
            upper_tree,
            mb_tree_fanout,
            db,
        }
    }

    pub fn insert(&mut self, key: AddrKey, version_id: u32, payload: StateValue) {
        let result = self.upper_tree.search(key);
        match result {
            Some(v) => {
                // old addr_key
                let pointer_hash = v.0;
                let tree_id = pointer_hash.pointer;
                let mut mbtree = PersistMBTree::open(tree_id, self.mb_tree_fanout, &self.db);
                merkle_btree_storage::insert(&mut mbtree, VersionKey(version_id), payload);
                let low_root_h = mbtree.get_root_hash();
                self.upper_tree.insert(key, tree_id, low_root_h);
            },
            None => {
                // new addr_key
                self.key_counter += 1;
                let tree_id = self.key_counter;
                let mut mbtree = PersistMBTree::new(tree_id, self.mb_tree_fanout, &self.db);
                merkle_btree_storage::insert(&mut mbtree, VersionKey(version_id), payload);
                let low_root_h = mbtree.get_root_hash();
                self.upper_tree.insert(key, tree_id, low_root_h);
            }
        }
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, (u32, StateValue)>) {
        let mut new_keys = BTreeMap::<AddrKey, Vec<(u32, StateValue)>>::new();
        let mut old_keys_and_search_results = BTreeMap::<AddrKey, Vec<(u64, u32, StateValue)>>::new();
        for (key, (version_id, payload)) in inputs {
            let result = self.upper_tree.search(key);
            match result {
                Some(v) => {
                    // old key
                    let pointer_hash = v.0;
                    let tree_id = pointer_hash.pointer;
                    if old_keys_and_search_results.contains_key(&key) {
                        let v = old_keys_and_search_results.get_mut(&key).unwrap();
                        v.push((tree_id, version_id, payload));
                    } else {
                        let v = vec![(tree_id, version_id, payload)];
                        old_keys_and_search_results.insert(key, v);
                    }
                },
                None => {
                    // new key
                    if new_keys.contains_key(&key) {
                        let v = new_keys.get_mut(&key).unwrap();
                        v.push((version_id, payload));
                    } else {
                        let v = vec![(version_id, payload)];
                        new_keys.insert(key, v);
                    }
                }
            }
        }

        let mut key_pointer_hash = BTreeMap::<AddrKey, (u64, H256)>::new();
        for (key, v) in old_keys_and_search_results {
            let tree_id = v[0].0;
            let mut mbtree = PersistMBTree::open(tree_id, self.mb_tree_fanout, &self.db);
            let mut mbtree_write_context = WriteContext::new(&mut mbtree);
            for (_, version_id, payload) in v {
                mbtree_write_context.insert(VersionKey(version_id), payload);
            }
            mbtree_write_context.persist();
            let low_root_h = mbtree.get_root_hash();
            key_pointer_hash.insert(key, (tree_id, low_root_h));
        }

        for (key, v) in new_keys {
            self.key_counter += 1;
            let tree_id = self.key_counter;
            let mut mbtree = PersistMBTree::new(tree_id, self.mb_tree_fanout, &self.db);
            let mut mbtree_write_context = WriteContext::new(&mut mbtree);
            for (version_id, payload) in v {
                mbtree_write_context.insert(VersionKey(version_id), payload);
            }
            mbtree_write_context.persist();
            let low_root_h = mbtree.get_root_hash();
            key_pointer_hash.insert(key, (tree_id, low_root_h));
        }

        self.upper_tree.batch_insert(key_pointer_hash);
    }

    pub fn get_root(&self) -> H256 {
        self.upper_tree.get_root()
    }

    pub fn search_with_latest_version(&self, key: AddrKey) -> Option<StateValue> {
        let r = self.upper_tree.search(key);
        match r {
            None => None,
            Some(b) => {
                let pointer_hash = b.0;
                let tree_id = pointer_hash.pointer;
                let mbtree = PersistMBTree::open(tree_id, self.mb_tree_fanout, &self.db);
                let (_, v) = merkle_btree_storage::get_right_most_data(&mbtree);
                return Some(v);
            }
        }
    }

    pub fn search_with_proof(&self, key: AddrKey, lb: u32, ub: u32) -> (Option<Vec<(VersionKey, StateValue)>>, TotalProof) {
        let (r, p) = self.upper_tree.search_with_proof(key);
        match r {
            None => {
                let total_proof = TotalProof {
                    upper_proof: (None, p),
                    lower_proof: None,
                };
                return (None, total_proof);
            },
            Some(b) => {
                let pointer_hash = b.0;
                let tree_id = pointer_hash.pointer;
                let low_h = pointer_hash.hash;
                let mbtree = PersistMBTree::open(tree_id, self.mb_tree_fanout, &self.db);
                let (read_v, mb_p) = merkle_btree_storage::get_range_proof(&mbtree, VersionKey(lb), VersionKey(ub));
                let total_proof = TotalProof {
                    upper_proof: (Some((tree_id, low_h)), p),
                    lower_proof: Some(mb_p),
                };
                return (read_v, total_proof);
            }
        }
    }
}

pub fn verify_non_learn_cmi(key: AddrKey, lb: u32, ub: u32, result: &Option<Vec<(VersionKey, StateValue)>>, digest: H256, proof: &TotalProof) -> bool {
    let mut error_flag = true;
    let upper_proof = &proof.upper_proof;
    let lower_proof = &proof.lower_proof;
    if result.is_none() {
        // no result
        if lower_proof.is_some() {
            error_flag = false;
        }
        let b = verify_with_addr_key(&key, None, digest, &upper_proof.1);
        if b == false {
            error_flag = false;
        }
    } else {
        // result exists
        if lower_proof.is_none() {
            error_flag = false;
        }

        if upper_proof.0.is_none() {
            error_flag = false;
        } else {
            let (tree_id, low_h) = upper_proof.0.unwrap();
            let lp = lower_proof.as_ref().unwrap();
            let h = merkle_btree_storage::reconstruct_range_proof(VersionKey(lb), VersionKey(ub), result, lp);
            if h != low_h {
                error_flag = false;
            }

            let upper_result = Some(Value(PointerHash {
                pointer: tree_id,
                hash: low_h,
            }));
            let b = verify_with_addr_key(&key, upper_result, digest, &upper_proof.1);
            if b == false {
                error_flag = false;
            }
        }
    }
    return error_flag;
}
#[cfg(test)]
mod tests {
    use utils::H160;
    use kvdb_rocksdb::DatabaseConfig;
    use rand::prelude::*;
    use super::*;
    #[test]
    fn test_non_learn_cmi() {
        let mut rng = StdRng::seed_from_u64(1);
        let path = "persist_trie";
        if std::path::Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        std::fs::create_dir(&path).unwrap_or_default();
        let mut db_config = DatabaseConfig::with_columns(1);
        db_config.memory_budget.insert(ROCKSDB_COL_ID, 64);
        let db = Database::open(&db_config, path).unwrap();
        let num_of_contract = 100;
        let num_of_addr = 100;
        let num_of_versions = 100;

        let fanout = 10;
        
        let mut addr_key_vec = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_addr {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let addr_key = AddrKey::new(acc_addr.into(), state_addr.into());
                addr_key_vec.push(addr_key);
            }
        }

        {
            let mut tree = NonLearnCMI::new(&db, fanout);
            let start = std::time::Instant::now();
            for version in 1..=num_of_versions {
                let mut inputs = BTreeMap::<AddrKey, (u32, StateValue)>::new();
                for addr_key in &addr_key_vec {
                    let value = StateValue(H256::from_low_u64_be((version * 2) as u64));
                    inputs.insert(*addr_key, (version, value));
                }
                tree.batch_insert(inputs);
            }
            /* for addr_key in &addr_key_vec {
                let mut inputs = BTreeMap::<AddrKey, (u32, StateValue)>::new();
                for version in 1..=num_of_versions {
                    let value = StateValue::new(H256::from_low_u64_be((version * 2) as u64));
                    inputs.insert(*addr_key, (version, value));
                }
                tree.batch_insert(inputs);
            } */
            let elapse = start.elapsed().as_nanos();
            println!("insertion: {:?}", elapse / (num_of_contract * num_of_addr * num_of_versions) as u128);
        }
        

        // let tree = NonLearnCMI::open(&db, fanout);
        // let start = std::time::Instant::now();
        // for addr in &addr_key_vec {
        //     let v = tree.search_with_latest_version(*addr).unwrap();
        //     let true_v = StateValue(H256::from_low_u64_be((2*num_of_versions) as u64));
        //     if v != true_v {
        //         println!("false, addr: {:?}, v: {:?}, true_v: {:?}", addr, v, true_v);
        //     }
        // }
        // let elapse = start.elapsed().as_nanos();
        // println!("search latest version: {:?}", elapse / (num_of_contract * num_of_addr) as u128);
        // let digest = tree.get_root();
        // let start = std::time::Instant::now();
        // for addr in &addr_key_vec {
        //     for version in 1..=num_of_versions {
        //         let (v, p) = tree.search_with_proof(*addr, version, version);
        //         let b = verify_non_learn_cmi(*addr, version, version, &v, digest, &p);
        //         if b == false {
        //             println!("false");
        //         }
        //         let current_v = StateValue(H256::from_low_u64_be((2*version) as u64));
        //         let read_v = v.unwrap()[0].1;
        //         if current_v != read_v {
        //             println!("key: {:?}, true v: {:?}, read_v: {:?}", addr, current_v, read_v);
        //         }
        //     }
        // }
        // let elapse = start.elapsed().as_nanos();
        // println!("search prove: {:?}", elapse / (num_of_contract * num_of_addr * num_of_versions) as u128);
    }
}
