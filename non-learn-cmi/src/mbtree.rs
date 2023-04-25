use kvdb_rocksdb::Database;
use serde::{Serialize, Deserialize};
use utils::{H256, types::{Digestible, bytes_hash, StateValue}, ROCKSDB_COL_ID};
use std::collections::BTreeMap;
use merkle_btree_storage::{traits::BPlusTreeNodeIO, nodes::BPlusTreeNode, get_left_most_leaf_id};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Serialize, Deserialize, Debug, Hash)]
pub struct VersionKey(pub u32);

impl Digestible for VersionKey {
    fn to_digest(&self) -> H256 {
        bytes_hash(&self.0.to_be_bytes())
    }
}

pub struct PersistMBTree<'a> {
    tree_id: u64,
    root: u32,
    counter: u32,
    key_num: u32,
    fanout: usize,
    db: &'a Database
}

impl<'a> PersistMBTree<'a> {
    pub fn new(tree_id: u64, fanout: usize, db: &'a Database) -> Self {
        Self {
            tree_id,
            root: 0,
            counter: 0,
            key_num: 0,
            fanout,
            db,
        }
    }

    pub fn open(tree_id: u64, fanout: usize, db: &'a Database) -> Self {
        let root_load_key = Self::get_root_storage_key(tree_id);
        let root: u32 = match db.get(ROCKSDB_COL_ID, &root_load_key).unwrap() {
            Some(v) => {
                u32::from_be_bytes(v.try_into().unwrap())
            },
            None => {
                0
            }
        };

        let counter_load_key = Self::get_counter_storage_key(tree_id);
        let counter: u32 = match db.get(ROCKSDB_COL_ID, &counter_load_key).unwrap() {
            Some(v) => {
                u32::from_be_bytes(v.try_into().unwrap())
            },
            None => {
                0
            }
        };

        let key_num_load_key = Self::get_key_num_storage_key(tree_id);
        let key_num: u32 = match db.get(ROCKSDB_COL_ID, &key_num_load_key).unwrap() {
            Some(v) => {
                u32::from_be_bytes(v.try_into().unwrap())
            },
            None => {
                0
            }
        };

        Self {
            tree_id,
            root,
            counter,
            key_num,
            fanout,
            db,
        }
    }

    pub fn get_load_key(&self, node_id: u32) -> Vec<u8> {
        let mut load_key = self.tree_id.to_be_bytes().to_vec();
        load_key.extend(&node_id.to_be_bytes());
        return load_key;
    }
}

impl<'a> BPlusTreeNodeIO<VersionKey, StateValue> for PersistMBTree<'a> {
    fn load_node(&self, node_id: u32) -> Option<BPlusTreeNode<VersionKey, StateValue>> {
        let load_key = self.get_load_key(node_id);
        let r = self.db.get(ROCKSDB_COL_ID, &load_key).unwrap();
        match r {
            Some(v) => {
                let node_bytes = v;
                let node: BPlusTreeNode<VersionKey, StateValue> = bincode::deserialize(&node_bytes).unwrap();
                Some(node)
            },
            None => {
                return None;
            }
        }
    }

    fn store_node(&mut self, node_id: u32, node: BPlusTreeNode<VersionKey, StateValue>) {
        let load_key = self.get_load_key(node_id);
        let node_bytes = bincode::serialize(&node).unwrap();
        let mut tx = self.db.transaction();
        tx.put(ROCKSDB_COL_ID, &load_key, &node_bytes);
        self.db.write(tx).unwrap();
    }

    fn store_nodes_batch(&mut self, nodes: &BTreeMap::<u32, BPlusTreeNode<VersionKey, StateValue>>) {
        let mut tx = self.db.transaction();
        for (node_id, node) in nodes {
            let load_key = self.get_load_key(*node_id);
            let node_bytes = bincode::serialize(node).unwrap();
            tx.put(ROCKSDB_COL_ID, &load_key, &node_bytes);
        }
        self.db.write(tx).unwrap();
    }

    fn new_counter(&mut self) -> u32 {
        self.counter += 1;
        return self.counter;
    }

    fn set_counter(&mut self, counter: u32) {
        self.counter = counter;
    }

    fn get_counter(&self) -> u32 {
        self.counter
    }

    fn get_root_id(&self) -> u32 {
        self.root
    }

    fn set_root_id(&mut self, root_id: u32) {
        self.root = root_id;
    }

    fn get_root_hash(&self) -> H256 {
        let root_node = self.load_node(self.get_root_id()).unwrap();
        return root_node.to_digest();
    }

    fn increment_key_num(&mut self) {
        self.key_num += 1;
    }

    fn set_key_num(&mut self, key_num: u32) {
        self.key_num = key_num;
    }

    fn get_key_num(&self) -> u32 {
        self.key_num
    }

    fn load_all_key_values(&self) -> Vec<(VersionKey, StateValue)> {
        let mut key_values = Vec::<(VersionKey, StateValue)>::new();
        let mut cur_leaf_id = get_left_most_leaf_id(self);
        while cur_leaf_id != 0 {
            let leaf = self.load_node(cur_leaf_id).unwrap().to_leaf().unwrap();
            for i in 0..leaf.get_n() {
                let key_value = leaf.key_values[i];
                key_values.push(key_value);
            }
            cur_leaf_id = leaf.next;
        }
        return key_values;
    }

    fn get_storage_id(&self,) -> u64 {
        self.tree_id
    }

    fn get_root_storage_key(id: u64) -> Vec<u8> {
        let mut root_load_key = "r".as_bytes().to_vec();
        root_load_key.extend(id.to_be_bytes());
        return root_load_key;
    }

    fn get_counter_storage_key(id: u64) -> Vec<u8> {
        let mut counter_load_key = "c".as_bytes().to_vec();
        counter_load_key.extend(id.to_be_bytes());
        return counter_load_key;
    }

    fn get_key_num_storage_key(id: u64) -> Vec<u8> {
        let mut key_num_load_key = "n".as_bytes().to_vec();
        key_num_load_key.extend(id.to_be_bytes());
        return key_num_load_key;
    }

    fn set_fanout(&mut self, fanout: usize) {
        self.fanout = fanout;
    }

    fn get_fanout(&self) -> usize {
        self.fanout
    }
}

impl<'a> Drop for PersistMBTree<'a> {
    // persist root, counter, and key_num of the mbtree
    fn drop(&mut self) {
        let mut tx = self.db.transaction();
        let root_load_key = Self::get_root_storage_key(self.tree_id);
        let root_bytes = self.root.to_be_bytes().to_vec();
        tx.put(ROCKSDB_COL_ID, &root_load_key, &root_bytes);

        let counter_load_key = Self::get_counter_storage_key(self.tree_id);
        let counter_bytes = self.counter.to_be_bytes().to_vec();
        tx.put(ROCKSDB_COL_ID, &counter_load_key, &counter_bytes);

        let key_num_load_key = Self::get_key_num_storage_key(self.tree_id);
        let key_num_bytes = self.key_num.to_be_bytes().to_vec();
        tx.put(ROCKSDB_COL_ID, &key_num_load_key, &key_num_bytes);

        self.db.write(tx).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use kvdb_rocksdb::DatabaseConfig;
    use merkle_btree_storage::{insert, get_range_proof, reconstruct_range_proof, get_right_most_data};
    use super::*;
    #[test]
    fn test_persist_mbtree() {
        let path = "persist_trie";
        if std::path::Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        std::fs::create_dir(&path).unwrap_or_default();
        let mut db_config = DatabaseConfig::with_columns(1);
        db_config.memory_budget.insert(ROCKSDB_COL_ID, 64);
        let db = Database::open(&db_config, path).unwrap();

        let n = 10000;
        let keys: Vec<VersionKey> = (0..n).map(|i| {
            VersionKey(i)
        }).collect();

        let values1: Vec<StateValue> = (0..n).map(|i| {
            StateValue(H256::from_low_u64_be(i as u64))
        }).collect();

        let values2: Vec<StateValue> = (0..n).map(|i| {
            StateValue(H256::from_low_u64_be(2*i as u64))
        }).collect();

        let fanout = 10;
        {
            let mut mbtree1 = PersistMBTree::new(1, fanout, &db);
            let mut mbtree2 = PersistMBTree::new(2, fanout, &db);
            let start = std::time::Instant::now();
            for (key, value) in keys.iter().zip(values1.iter()) {
                insert(&mut mbtree1, *key, *value);
            }
    
            for (key, value) in keys.iter().zip(values2.iter()) {
                insert(&mut mbtree2, *key, *value);
            }
            let elapse = start.elapsed().as_nanos();
            println!("average insert: {}", elapse / n as u128 / 2);
        }

        let mbtree1 = PersistMBTree::open(1, fanout, &db);
        let mbtree2 = PersistMBTree::open(2, fanout, &db);
        let root_hash1 = mbtree1.get_root_hash();
        let root_hash2 = mbtree2.get_root_hash();

        let start = std::time::Instant::now();
        for (key, value) in keys.iter().zip(values1.iter()) {
            let (v, proof) = get_range_proof(&mbtree1, *key, *key);
            let r = v.as_ref().unwrap()[0].1;
            if &r != value {
                println!("false");
                println!("key: {:?}, value: {:?}", key, value);
            } 
            let h = reconstruct_range_proof(*key, *key, &v, &proof);
            if h != root_hash1 {
                println!("false");
                println!("key: {:?}, value: {:?}", key, value);
            }
        }

        for (key, value) in keys.iter().zip(values2.iter()) {
            let (v, proof) = get_range_proof(&mbtree2, *key, *key);
            let r = v.as_ref().unwrap()[0].1;
            if &r != value {
                println!("false");
                println!("key: {:?}, value: {:?}", key, value);
            } 
            let h = reconstruct_range_proof(*key, *key, &v, &proof);
            if h != root_hash2 {
                println!("false");
                println!("key: {:?}, value: {:?}", key, value);
            }
        }

        let elapse = start.elapsed().as_nanos();
        println!("average query: {}", elapse / n as u128 / 2);

        println!("key_num1: {}", mbtree1.get_key_num());
        println!("key_num2: {}", mbtree2.get_key_num());

        let start = std::time::Instant::now();
        let (key1, value1) = get_right_most_data(&mbtree1);
        let elapse = start.elapsed().as_nanos();
        println!("latest query 1: {}", elapse);
        let start = std::time::Instant::now();
        let (key2, value2) = get_right_most_data(&mbtree2);
        let elapse = start.elapsed().as_nanos();
        println!("latest query 2: {}", elapse);
        let true_value_1 = StateValue(H256::from_low_u64_be((n - 1) as u64));
        assert_eq!(value1, true_value_1);
        let true_value_2 = StateValue(H256::from_low_u64_be(2*(n - 1) as u64));
        assert_eq!(value2, true_value_2);
        println!("k1 {:?} v1 {:?} k2 {:?} v2 {:?}", key1, value1, key2, value2);
    }
}