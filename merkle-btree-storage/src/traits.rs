use std::collections::BTreeMap;
use utils::{types::{Num, Value}, H256};
use crate::nodes::BPlusTreeNode;

/* Trait of the Merkle B+-Tree Storage
 */
pub trait BPlusTreeNodeIO<K: Num, V: Value> {
    // given a node id, load a node from the storage, if the id does not exist, return none
    fn load_node(&self, node_id: u32) -> Option<BPlusTreeNode<K, V>>;
    // store a node with node_id
    fn store_node(&mut self, node_id: u32, node: BPlusTreeNode<K, V>);
    // store a batch of nodes in a map
    fn store_nodes_batch(&mut self, nodes: &BTreeMap::<u32, BPlusTreeNode<K, V>>);
    // create a new counter of the node_id
    fn new_counter(&mut self) -> u32;
    // set the latest counter
    fn set_counter(&mut self, counter: u32);
    // get the latest counter of the storage
    fn get_counter(&self) -> u32;
    // get the id of the root node
    fn get_root_id(&self) -> u32;
    // set the id of the root node as root_id
    fn set_root_id(&mut self, root_id: u32);
    // get the root hash of the storage
    fn get_root_hash(&self) -> H256;
    // increment the number of keys in the storage
    fn increment_key_num(&mut self);
    // set the number of keys in the storage
    fn set_key_num(&mut self, key_num: u32);
    // get the number of keys in the storage
    fn get_key_num(&self) -> u32;
    // batchly load all the key-value pairs from the storage
    fn load_all_key_values(&self) -> Vec<(K, V)>;
    // get the id of the storage (used for multiple MB-Tree in the same database storage)
    fn get_storage_id(&self,) -> u64;
    // get the storage key of the root (used for persisting the root digest of the storage to the database)
    fn get_root_storage_key(id: u64) -> Vec<u8>;
    // get the storage key of the counter
    fn get_counter_storage_key(id: u64) -> Vec<u8>;
    // get the storage key of the number of data in the storage
    fn get_key_num_storage_key(id: u64) -> Vec<u8>;
    // set the fanout of the MB-Tree
    fn set_fanout(&mut self, fanout: usize);
    // get the fanout of the MB-Tree
    fn get_fanout(&self) -> usize;
}