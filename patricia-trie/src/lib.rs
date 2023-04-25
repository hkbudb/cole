pub(crate) mod hash;
pub mod nibbles;
pub mod proof;
pub mod read;
pub mod storage;
pub mod traits;
pub mod u4;
pub mod write;

pub mod prelude;
use anyhow::Context;
use kvdb_rocksdb::{Database};
pub use prelude::*;
use serde::{Deserialize, Serialize};
use utils::{types::{StateValue, AddrKey}, ROCKSDB_COL_ID};
use std::collections::{HashMap, BTreeMap};
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Key(pub NibbleBuf);

impl AsNibbles for Key {
    fn as_nibbles(&self) -> Nibbles<'_> {
        self.0.as_nibbles()
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct TestTrie {
    roots: Vec<H256>,
    nodes: HashMap<Vec<u8>, Vec<u8>>,
}

impl NodeLoader<StateValue> for TestTrie {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let id_bytes = id.as_bytes().to_vec();
        let node_bytes = self.nodes.get(&id_bytes).cloned().context("Unknown node")?;
        let node: TrieNode<StateValue> = bincode::deserialize(&node_bytes).unwrap();
        Ok(node)
    }
}

impl NodeLoader<StateValue> for &'_ TestTrie {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let id_bytes = id.as_bytes().to_vec();
        let node_bytes = self.nodes.get(&id_bytes).cloned().context("Unknown node")?;
        let node: TrieNode<StateValue> = bincode::deserialize(&node_bytes).unwrap();
        Ok(node)
    }
}

impl TestTrie {
    pub fn new() ->  Self {
        Self {
            roots: vec![H256::default()],
            nodes: HashMap::new(),
        }
    }
    #[allow(dead_code)]
    pub fn apply(&mut self, apply: Apply<StateValue>) {
        self.roots.push(apply.root);
        let mut ser_nodes = HashMap::new();
        for (h, node) in apply.nodes.into_iter() {
            let h_bytes = h.as_bytes().to_vec();
            let node_bytes = bincode::serialize(&node).unwrap();
            ser_nodes.insert(h_bytes, node_bytes);
        }
        self.nodes.extend(ser_nodes);
    }

    pub fn get_latest_root(&self) -> H256 {
        let l = self.roots.len() - 1;
        self.roots[l]
    }

    pub fn get_roots_len(&self) -> usize {
        self.roots.len()
    }

    pub fn get_root_with_version(&self, version: u32) -> H256 {
        // version starts with 1
        match self.roots.get(version as usize) {
            Some(r) => *r,
            None => H256::default(),
        }
    }

    pub fn search(&self, addr_key: AddrKey) -> Option<StateValue> {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let read_v = read_trie_without_proof(&self, self.get_latest_root(), &k).unwrap();
        return read_v;
    }

    pub fn search_with_proof(&self, addr_key: AddrKey, version: u32) -> (Option<StateValue>, Proof) {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let (read_v, proof) = read_trie(&self, self.get_root_with_version(version), &k).unwrap();
        return (read_v, proof);
    }

    pub fn insert(&mut self, addr_key: AddrKey, value: StateValue) {
        let immut_ref = unsafe {
            (self as *const TestTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        ctx.insert(&k, value).unwrap();
        let changes = ctx.changes();
        self.apply(changes);
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, StateValue>) {
        let immut_ref = unsafe {
            (self as *const TestTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        for (addr_key, value) in inputs {
            let k = Key(NibbleBuf::from_addr_key(addr_key));
            ctx.insert(&k, value).unwrap();
        }
        let changes = ctx.changes();
        self.apply(changes);
    }
}

pub struct PersistTrie<'a> {
    pub roots: Vec<H256>,
    pub db: &'a Database
}

impl<'a> Drop for PersistTrie<'a> {
    // persist the meta data including the len of roots and hashes in roots
    fn drop(&mut self) {
        let meta_key = "meta".as_bytes();
        let roots_len = self.roots.len() as u32;
        let mut meta_bytes = roots_len.to_be_bytes().to_vec();
        for root in &self.roots {
            meta_bytes.extend(root.as_bytes());
        }
        let mut tx = self.db.transaction();
        tx.put(ROCKSDB_COL_ID, meta_key, &meta_bytes);
        self.db.write(tx).unwrap();
    }
}

impl<'a> PersistTrie<'a> {
    // init a persist trie with the db reference
    pub fn new(db: &'a Database) -> Self {
        Self {
            roots: vec![H256::default()],
            db,
        }
    }
    // load the trie from the db
    pub fn open(db: &'a Database) -> Self {
        let mut roots = Vec::<H256>::new();
        let meta_key = "meta".as_bytes();
        match db.get(ROCKSDB_COL_ID, meta_key).unwrap() {
            Some(r) => {
                // load the len of hash vec
                let len = u32::from_be_bytes(r[0..4].try_into().unwrap()) as usize;
                for i in 0..len {
                    let start_idx = 4 + i * 32;
                    let h = H256::from_slice(&r[start_idx .. start_idx+32]);
                    roots.push(h);
                }
            },
            None => {}
        }
        Self {
            roots,
            db,
        }
    }

    pub fn apply(&mut self, apply: Apply<StateValue>) {
        self.roots.push(apply.root);
        let mut tx = self.db.transaction();
        for (k, v) in apply.nodes {
            let v_bytes = bincode::serialize(&v).unwrap();
            tx.put(ROCKSDB_COL_ID, &k.as_bytes(), &v_bytes);
        }
        self.db.write(tx).unwrap();
    }

    pub fn get_latest_root(&self) -> H256 {
        let l = self.roots.len() - 1;
        self.roots[l]
    }

    pub fn get_roots_len(&self) -> usize {
        self.roots.len()
    }

    pub fn get_root_with_version(&self, version: u32) -> H256 {
        // version starts with 1
        match self.roots.get(version as usize) {
            Some(r) => *r,
            None => H256::default(),
        }
    }

    pub fn search(&self, addr_key: AddrKey) -> Option<StateValue> {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let read_v = read_trie_without_proof(&self, self.get_latest_root(), &k).unwrap();
        return read_v;
    }

    pub fn search_with_proof(&self, addr_key: AddrKey, version: u32) -> (Option<StateValue>, Proof) {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let (read_v, proof) = read_trie(&self, self.get_root_with_version(version), &k).unwrap();
        return (read_v, proof);
    }

    pub fn insert(&mut self, addr_key: AddrKey, value: StateValue) {
        let immut_ref = unsafe {
            (self as *const PersistTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        ctx.insert(&k, value).unwrap();
        let changes = ctx.changes();
        self.apply(changes);
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, StateValue>) {
        let immut_ref = unsafe {
            (self as *const PersistTrie).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_latest_root());
        for (addr_key, value) in inputs {
            let k = Key(NibbleBuf::from_addr_key(addr_key));
            ctx.insert(&k, value).unwrap();
        }
        let changes = ctx.changes();
        self.apply(changes);
    }
}

impl NodeLoader<StateValue> for PersistTrie<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<StateValue> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

impl NodeLoader<StateValue> for &'_ PersistTrie<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<StateValue>> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<StateValue> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

pub fn verify_trie_proof(key: &Key, value: Option<StateValue>, root_h: H256, proof: &Proof) -> bool {
    let mut error_flag = true;
    if root_h != proof.root_hash() {
        error_flag = false;
    }

    match value {
        Some(v) => {
            if proof.value_hash(key) != Some(v.to_digest()) {
                error_flag = false;
            }
        },
        None => {
            if proof.value_hash(key) != Some(H256::default()) {
                error_flag = false;
            }
        }
    }

    return error_flag;
}

pub fn verify_with_addr_key(addr_key: &AddrKey, value: Option<StateValue>, root_h: H256, proof: &Proof) -> bool {
    let key = Key(NibbleBuf::from_addr_key(*addr_key));
    let mut error_flag = true;
    if root_h != proof.root_hash() {
        error_flag = false;
    }

    match value {
        Some(v) => {
            if proof.value_hash(&key) != Some(v.to_digest()) {
                error_flag = false;
            }
        },
        None => {
            if proof.value_hash(&key) != Some(H256::default()) {
                error_flag = false;
            }
        }
    }

    return error_flag;
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use utils::{types::AddrKey, H160};
    use std::path::Path;
    use kvdb_rocksdb::DatabaseConfig;
    use super::*;
    
    #[test]
    fn test_trie_in_memory() {
        let num_of_contract = 100;
        let num_of_address = 100;
        let num_of_versions = 200;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }

        let mut trie = TestTrie::new();
        for i in 1..=num_of_versions {
            let v = StateValue(H256::from_low_u64_be(i));
            let mut map = BTreeMap::new();
            for key in &keys {
                map.insert(*key, v);
            }
            trie.batch_insert(map);
        }
        println!("finish insert");
        let latest_v = StateValue(H256::from_low_u64_be(num_of_versions));
        for key in &keys {
            let v = trie.search(*key).unwrap();
            assert_eq!(v, latest_v);
            for i in 1..=num_of_versions {
                let (v, p) = trie.search_with_proof(*key, i as u32);
                let current_v = StateValue(H256::from_low_u64_be(i));
                assert_eq!(current_v, v.unwrap());
                let b = verify_with_addr_key(key, v, trie.get_root_with_version(i as u32), &p);
                assert!(b);
            }
        }
    }

    #[test]
    fn test_disk_trie() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_config = DatabaseConfig::with_columns(1);
        db_config.memory_budget.insert(ROCKSDB_COL_ID, 64);
        db_config.memory_budget.insert(1, 64);
        let db = Database::open(&db_config, path).unwrap();
        let num_of_contract = 100;
        let num_of_address = 100;
        let num_of_versions = 100;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }

        {
            let mut trie = PersistTrie::new(&db);
            let start = std::time::Instant::now();
            for i in 1..=num_of_versions {
                let v = StateValue(H256::from_low_u64_be(i));
                let mut map = BTreeMap::new();
                for key in &keys {
                    map.insert(*key, v);
                }
                trie.batch_insert(map);
            }
            let elapse = start.elapsed().as_nanos();
            println!("avg insert time: {}", elapse / (num_of_address * num_of_contract * num_of_versions) as u128);
            println!("finish insert");
        }
        
/*         let trie = PersistTrie::open(&db);
        let mut search_latest = 0;
        let mut search_prove = 0;
        let latest_v = StateValue(H256::from_low_u64_be(num_of_versions));
        for key in &keys {
            let start = std::time::Instant::now();
            let v = trie.search(*key).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(v, latest_v);
            for i in 1..=num_of_versions {
                let start = std::time::Instant::now();
                let (v, p) = trie.search_with_proof(*key, i as u32);
                let b = verify_with_addr_key(key, v, trie.get_root_with_version(i as u32), &p);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                let current_v = StateValue(H256::from_low_u64_be(i));
                let read_v = v.unwrap();
                if current_v != read_v {
                    println!("key: {:?}, true v: {:?}, read_v: {:?}", key, current_v, read_v);
                }
                // assert_eq!(current_v, v.unwrap());
                assert!(b);
            }
        }
        println!("search latest: {}", search_latest / (num_of_address * num_of_contract) as u128);
        println!("search prove: {}", search_prove / (num_of_address * num_of_contract * num_of_versions) as u128); */
    }
}