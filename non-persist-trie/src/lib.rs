pub use patricia_trie::prelude::*;
use serde::{Deserialize, Serialize};
use kvdb_rocksdb::Database;
use std::collections::{HashSet, BTreeMap};
use utils::{types::{AddrKey, Digestible, bytes_hash}, ROCKSDB_COL_ID};
use patricia_trie::Key;

/* Store the pointer and the root digest of the lower tree
 */
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointerHash {
    pub pointer: u64,
    pub hash: H256,
}

impl Digestible for PointerHash {
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        bytes.extend(&self.pointer.to_be_bytes());
        bytes.extend(self.hash.as_bytes());
        bytes_hash(&bytes)
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct Value(pub PointerHash);

impl Digestible for Value {
    fn to_digest(&self) -> H256 {
        self.0.to_digest()
    }
}

pub struct NonPersistMPT<'a> {
    root: H256,
    pub db: &'a Database,
}

impl<'a> Drop for NonPersistMPT<'a> {
    fn drop(&mut self) {
        let root_bytes = self.root.as_bytes().to_vec();
        let mut tx = self.db.transaction();
        let root_get_key = "mptroot".as_bytes().to_vec();
        tx.put(ROCKSDB_COL_ID, &root_get_key, &root_bytes);
        self.db.write(tx).unwrap();
    }
}

impl<'a> NonPersistMPT<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self {
            root: H256::default(),
            db,
        }
    }

    pub fn open(db: &'a Database) -> Self {
        let root_get_key = "mptroot".as_bytes().to_vec();
        let root_bytes = db.get(ROCKSDB_COL_ID, &root_get_key).unwrap().unwrap();
        let root = H256::from_slice(&root_bytes);
        Self {
            root,
            db,
        }
    }

    pub fn apply(&mut self, apply: Apply<Value>) {
        self.root = apply.root;
        let mut tx = self.db.transaction();
        for (h, node) in apply.nodes {
            let h_bytes = h.as_bytes().to_vec();
            let bytes = bincode::serialize(&node).unwrap();
            tx.put(ROCKSDB_COL_ID, &h_bytes, &bytes);
        }
        self.db.write(tx).unwrap();
    }

    pub fn delete_nodes(&mut self, deletes: HashSet<H256>) {
        let mut tx = self.db.transaction();
        for h in deletes {
            let h_bytes = h.as_bytes().to_vec();
            tx.delete(ROCKSDB_COL_ID, &h_bytes);
        }
        self.db.write(tx).unwrap();
    }

    pub fn apply_and_delete(&mut self, apply: Apply<Value>, deletes: HashSet<H256>) {
        self.root = apply.root;
        let mut tx = self.db.transaction();
        for (h, node) in apply.nodes {
            let h_bytes = h.as_bytes().to_vec();
            let bytes = bincode::serialize(&node).unwrap();
            tx.put(ROCKSDB_COL_ID, &h_bytes, &bytes);
        }

        for h in deletes {
            let h_bytes = h.as_bytes().to_vec();
            tx.delete(ROCKSDB_COL_ID, &h_bytes);
        }

        self.db.write(tx).unwrap();
    }

    pub fn node_count(&self) -> u64 {
        self.db.num_keys(ROCKSDB_COL_ID).unwrap()
    }

    pub fn get_root(&self) -> H256 {
        self.root
    }

    pub fn search(&self, addr_key: AddrKey) -> Option<Value> {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let read_v = read_trie_without_proof(&self, self.get_root(), &k).unwrap();
        return read_v;
    }

    pub fn search_with_proof(&self, addr_key: AddrKey) -> (Option<Value>, Proof) {
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let (read_v, proof) = read_trie(&self, self.get_root(), &k).unwrap();
        return (read_v, proof);
    }

    pub fn insert(&mut self, addr_key: AddrKey, key_id: u64, h: H256) {
        let immut_ref = unsafe {
            (self as *const NonPersistMPT).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_root());
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        let value = Value(PointerHash { pointer: key_id, hash: h });
        ctx.insert(&k, value).unwrap();
        let deletes = ctx.deletes();
        let changes = ctx.changes();
        self.apply_and_delete(changes, deletes);
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, (u64, H256)>) {
        let immut_ref = unsafe {
            (self as *const NonPersistMPT).as_ref().unwrap()
        };
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(immut_ref, self.get_root());
        for (k, (pointer, h)) in inputs {
            let key = Key(NibbleBuf::from_addr_key(k));
            let value = Value(PointerHash { pointer: pointer, hash: h });
            ctx.insert(&key, value).unwrap();
        }
        let deletes = ctx.deletes();
        let changes = ctx.changes();
        self.apply_and_delete(changes, deletes);
    }
}

impl NodeLoader<Value> for NonPersistMPT<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<Value>> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<Value> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

impl NodeLoader<Value> for &'_ NonPersistMPT<'_> {
    fn load_node(&self, id: H256) -> Result<TrieNode<Value>> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: TrieNode<Value> = bincode::deserialize(&node_byte).unwrap();
        Ok(node)
    }
}

pub fn verify_with_addr_key(addr_key: &AddrKey, value: Option<Value>, root_h: H256, proof: &Proof) -> bool {
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
            if proof.value_hash(&key) != None {
                error_flag = false;
            }
        }
    }

    return error_flag;
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use std::path::Path;
    use utils::H160;
    use kvdb_rocksdb::DatabaseConfig;
    use super::*;
    #[test]
    fn test_non_persist_trie() {
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
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys_values = BTreeMap::<AddrKey, (u64, H256)>::new();
        for i in 1..=num_of_contract {
            for j in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                let key_id = i + j;
                let h = H256::from_low_u64_be(i + j);
                keys_values.insert(addr_key, (key_id, h));
            }
        }

        {
            let mut trie = NonPersistMPT::new(&db);
            let start = std::time::Instant::now();
            for (key, (key_id, h)) in &keys_values {
                trie.insert(*key, *key_id, *h);
            }
            let elapse = start.elapsed().as_nanos();
            println!("insert time: {}", elapse / (num_of_address * num_of_contract) as u128);
            println!("node count: {}", trie.node_count());
        }

        let trie = NonPersistMPT::open(&db);
        for (key, (key_id, h)) in &keys_values {
            let v = trie.search(*key).unwrap();
            let true_v = Value(PointerHash { pointer: *key_id, hash: *h });
            assert_eq!(v, true_v);
            let (v, p) = trie.search_with_proof(*key);
            assert_eq!(v.unwrap(), true_v);
            let b = verify_with_addr_key(key, v, trie.root, &p);
            assert!(b);
        }
    }
}
