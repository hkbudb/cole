pub mod traits;
pub mod nodes;
use utils::{types::{Num, Value, Digestible}, H256, anyhow, Result};
use traits::BPlusTreeNodeIO;
use serde::{Serialize, Deserialize};
use nodes::{BPlusTreeNode, BPlusTreeLeafNode, BPlusTreeInternalNode};
use std::collections::BTreeMap;

// get the left mode leaf's id for traversing all the leaf nodes to collect the data in batch
pub fn get_left_most_leaf_id<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>) -> u32 {
    // get root id
    let mut node_id = tree.get_root_id();
    // retrieve root node
    let mut node = tree.load_node(node_id).unwrap();
    // iteratively traverse the tree along the left-most sub-path
    while node.is_leaf() == false {
        let internal = node.get_internal().unwrap();
        // get the left-most child node id
        node_id = *internal.childs.first().unwrap();
        node = tree.load_node(node_id).unwrap();
    }
    return node_id;
}

// get the right most key-value pair
pub fn get_right_most_data<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>) -> (K, V) {
    // retrieve the root node
    let mut node_id = tree.get_root_id();
    let mut node = tree.load_node(node_id).unwrap();
    // iteratively traverse the right-most nodes for each level
    while node.is_leaf() == false {
        let internal = node.get_internal().unwrap();
        node_id = *internal.childs.last().unwrap();
        node = tree.load_node(node_id).unwrap();
    }
    let leaf = node.to_leaf().unwrap();
    let (key, value) = leaf.key_values.last().unwrap().clone();
    return (key, value);
}

// a search algorithm that only returns the search result, either Some(r) or None
pub fn search_without_proof<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>, key: K) -> Option<(K, V)> {
    let mut node_id = tree.get_root_id();
    if let Some(n) = tree.load_node(node_id) {
        let mut node = n;
        let mut value = V::default();
        let mut result_key = K::default();
        // iteratively traverse the tree using key, until a leaf is reached
        while node.is_leaf() != true {
            let index = node.search_prove_idx(key);
            node_id = node.get_internal_child(index).unwrap();
            node = tree.load_node(node_id).unwrap();
        }
        // a leaf node
        if node.is_leaf() == true {
            let mut index = node.search_prove_idx(key);
            let leaf = node.get_leaf().unwrap();
            // last index is found
            if index == leaf.get_n() {
                // out of bound
                index -= 1;
            }

            result_key = leaf.key_values[index].0;
            value = leaf.key_values[index].1;
        }
        return Some((result_key, value));
    }
    else {
        return None;
    }
}

/* Proof of a range query, each level consist of a vector of MB-Tree nodes
 */
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct RangeProof<K: Num, V: Value> {
    pub levels: Vec<Vec<((usize, usize), BPlusTreeNode<K, V>)>>, // the first two usize store the start_idx and end_idx of the searched entries in the node
}

pub fn get_range_proof<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>, lb: K, ub: K) -> (Option<Vec<(K, V)>>, RangeProof<K, V>) {
    // init a range proof
    let mut proof = RangeProof::default();
    if tree.get_key_num() == 0 {
        // tree is empty
        return (None, proof);
    } else {
        // load root node
        let node = tree.load_node(tree.get_root_id()).unwrap();
        // init a result value vector
        let mut value_vec = Vec::<(K, V)>::new();
        // create a queue to help traverse the tree
        let mut queue = Vec::<BPlusTreeNode<K, V>>::new();
        // push the root node to the queue
        queue.push(node);
        // some counter to help determine the number of nodes in the level
        let mut prev_cnt = 1;
        let mut cur_cnt = 0;
        // a temporary proof for the current level
        let mut cur_level_proof = Vec::<((usize, usize), BPlusTreeNode<K, V>)>::new();
        // traverse the tree in a while loop until the queue is empty
        while !queue.is_empty() {
            let cur_node = queue.remove(0);
            prev_cnt -= 1; // decrease the node counter of the previous level
            if !cur_node.is_leaf() {
                // the node is an internal node, retrieve the reference of the internal node
                let internal = cur_node.get_internal().unwrap();
                // given the lb and ub, get the position range of the child nodes
                let (start_idx, end_idx) = cur_node.search_prove_idx_range(lb, ub);
                // update the node counter for the level
                cur_cnt += end_idx - start_idx + 1;
                // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                // add the corresponding child nodes to the queue
                for idx in start_idx ..=end_idx {
                    let id = internal.childs[idx];
                    let child_node = tree.load_node(id).unwrap();
                    queue.push(child_node);
                }
            } else {
                // the node is a leaf node, retrieve the reference of the leaf node
                let leaf = cur_node.get_leaf().unwrap();
                // get the position range of the leaf node
                let (start_idx, end_idx) = cur_node.search_prove_idx_range(lb, ub);
                // update the node counter for the level
                cur_cnt += end_idx - start_idx + 1;
                // add the cur_node to the proof as well as the starting and ending position of the traversed entries
                cur_level_proof.push(((start_idx, end_idx), cur_node.clone()));
                // add the corresponding searched entries to the value_vec
                for id in start_idx ..= end_idx {
                    let key_value = leaf.key_values[id].clone();
                    value_vec.push(key_value);
                }
            }

            if prev_cnt == 0 {
                // if prev_cnt = 0, start a new level by assigning cur_cnt to prev_cnt and reset cur_cnt to 0
                prev_cnt = cur_cnt;
                cur_cnt = 0;
                // add the temporary proof of the current level to the final proof
                proof.levels.push(cur_level_proof.clone());
                cur_level_proof.clear();
            }
        }
        return (Some(value_vec), proof);
    }
}

// reconstruct the range proof to the root digest 
pub fn reconstruct_range_proof<K: Num, V: Value>(lb: K, ub: K, result: &Option<Vec<(K, V)>>, proof: &RangeProof<K, V>) -> H256 {
    if result.is_none() && proof == &RangeProof::default() {
        return H256::default();
    } else {
        // a flag to determine whethere there is an verification error
        let mut validate = true;
        // the root hash from the proof should be the first level's single node's digest
        let compute_root_hash = proof.levels[0][0].1.to_digest();
        // a temporary vector to store the hash values of the next level
        let mut next_level_hashes= Vec::<H256>::new();
        // iterate each of the levels in the proof
        for (i, level_proof) in proof.levels.iter().enumerate() {
            // check whether the hash valeus in the next_level_hashes vector (constructed during the prevous level) match the re-computed one for the current level or not
            if i != 0 {
                let mut computed_hashes = Vec::<H256>::new();
                for (_, cur_level_node) in level_proof {
                    computed_hashes.push(cur_level_node.to_digest());
                }
                if computed_hashes != next_level_hashes {
                    // not match
                    validate = false;
                    break;
                }
                // start another level by clearing the hashes
                next_level_hashes.clear();
            }
            // id of the result in the vector of the proof
            let mut leaf_id: usize = 0;
            for inner_proof in level_proof {
                // retrieve the node reference from the level proof
                let node = &inner_proof.1;
                // retrieve the start and end positions from the level proof
                let (start_idx, end_idx) = &inner_proof.0;
                if !node.is_leaf() {
                    // node is an internal node
                    let internal = node.get_internal().unwrap();
                    // add the hash values of the traversed child nodes to the next_level_hashes
                    for id in *start_idx..= *end_idx {
                        let h = internal.child_hashes[id];
                        next_level_hashes.push(h);
                    }
                } else {
                    // node is a leaf node
                    let result = result.as_ref().unwrap();
                    let leaf = node.get_leaf().unwrap();
                    // check the values in the result vector against the values in the proof
                    for (i, id) in (*start_idx..= *end_idx).into_iter().enumerate() {
                        let key_value = leaf.key_values[id];
                        if i == 0 {
                            if result[leaf_id] != key_value {
                                validate = false;
                                break;
                            }
                        } else {
                            let k = key_value.0;
                            if k < lb || k > ub || result[leaf_id] != key_value {
                                validate = false;
                                break;
                            }
                        }
                        leaf_id += 1;
                    }
                }
            }
        }
        if validate == false {
            return H256::default();
        }
        return compute_root_hash;
    }
}

// the followings are the insertion related code
// insert the key-value pair to the leaf, fanout is also given in the function input
pub fn insert_in_leaf_node<K: Num, V: Value>(leaf: BPlusTreeLeafNode<K, V>, key: K, value: V, fanout: usize) -> Result<BPlusTreeLeafNode<K, V>> {
    let mut leaf = leaf;
    if leaf.get_n() >= fanout {
        return Err(anyhow!("exceed the fanout"));
    } else {
        let index = leaf.search_insert_idx(key);
        leaf.key_values.insert(index, (key, value));
        return Ok(leaf);
    }
}
#[derive(Debug)]
pub struct SearchNodeId {
    pub node_id: u32, // searched node's id
    pub child_idx: Option<usize>, // the child node's position of the searched node; if the node is an internal node, then it is Some(idx), otherwise, it is None
}

// the context of the write operation (help to batchly insert the key-value pairs)
pub struct WriteContext<'a, K: Num, V: Value, T: BPlusTreeNodeIO<K, V>> {
    pub root_id: u32, // id of the root node
    pub counter: u32, // counter of the node id
    pub key_num: u32, // number of key-value pairs in the storage
    pub nodes: BTreeMap<u32, BPlusTreeNode<K, V>>, // storage of the nodes
    pub tree: &'a mut T, // the reference of the MB-Tree
}

impl<'a, K, V, T> WriteContext<'a, K, V, T> where K: Num, V: Value, T: BPlusTreeNodeIO<K, V> {
    // create a new context given the tree's reference
    pub fn new(tree: &'a mut T) -> Self {
        let root_id = tree.get_root_id();
        let counter = tree.get_counter();
        let key_num = tree.get_key_num();
        Self {
            root_id,
            counter,
            key_num,
            nodes: BTreeMap::new(),
            tree,
        }
    }

    // load a node given the node_id
    fn load_node(&self, node_id: u32) -> Option<BPlusTreeNode<K, V>> {
        if self.nodes.contains_key(&node_id) {
            // find node in write context nodes
            return Some(self.nodes.get(&node_id).unwrap().clone());
        } else {
            // find node in tree
            return self.tree.load_node(node_id);
        }
    }
    // get the root hash of the MB-Tree
    pub fn get_root_hash(&self) -> H256 {
        // first retrieve the root node and then compute the digest
        let root_node = self.load_node(self.root_id).unwrap();
        root_node.to_digest()
    }

    // get the fanout of the MB-Tree
    pub fn get_fanout(&self) -> usize {
        self.tree.get_fanout()
    }

    // store a node to the context's map
    fn store_node(&mut self, node_id: u32, node: BPlusTreeNode<K, V>) {
        self.nodes.insert(node_id, node);
    }

    // create a new counter by increment it
    fn new_counter(&mut self) -> u32 {
        self.counter += 1;
        self.counter
    }

    // get and set the root id
    fn get_root_id(&self) -> u32 {
        self.root_id
    }

    fn set_root_id(&mut self, root_id: u32) {
        self.root_id = root_id;
    }

    // increment the number of keys in the storage
    fn increment_key_num(&mut self,) {
        self.key_num += 1;
    }

    // given the node_id and the search key, get the node ids of the search path
    pub fn get_search_path_ids(&self, node_id: u32, key: K) -> Option<Vec<SearchNodeId>> {
        let mut cur_id = node_id;
        let curnode = self.load_node(node_id);
        match curnode {
            Some(node) => {
                let mut node = node;
                let mut node_ids = Vec::new();
                // iterative until the node is a leaf node
                while !node.is_leaf() {
                    let internal_node = node.get_internal().unwrap();
                    let index = internal_node.search_key_idx(key);
                    // node id vec consists of Some(index) of the child node, and the current node id
                    let search_node_id = SearchNodeId {
                        node_id: cur_id,
                        child_idx: Some(index),
                    };
                    node_ids.push(search_node_id);
                    // update cur_id to the child node's id
                    cur_id = internal_node.childs[index];
                    // update the node to the child node
                    node = self.load_node(cur_id).unwrap();
                }
                let search_node_id = SearchNodeId {
                    node_id: cur_id,
                    child_idx: None,
                };
                node_ids.push(search_node_id);
                return Some(node_ids);
            },
            None => {
                return None; // no node exist under node_id
            }
        }
    }

    // given a search key, return None if the root does not exist; return Some(key_exist, leaf_node_id) otherwise
    fn search_key(&self, key: K) -> Option<(bool, Vec<SearchNodeId>)> {
        // try to get the search path from the root_id
        let r = self.get_search_path_ids(self.get_root_id(), key);
        match r {
            Some(search_path) => {
                let leaf_node_id = search_path.last().unwrap().node_id;
                let leaf = self.load_node(leaf_node_id).unwrap().to_leaf().unwrap();
                for (i, k_v) in leaf.key_values.iter().enumerate() {
                    if key == k_v.0 {
                        // find the key, then update the index of the last node and return true
                        let mut node_ids = search_path;
                        let last = node_ids.last_mut().unwrap();
                        last.child_idx = Some(i);
                        return Some((true, node_ids));
                    }
                }
                // does not find the key, then return false
                return Some((false, search_path));
            },
            None => {
                return None;
            }
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        // add 1 to the number of data
        self.increment_key_num();
        let fanout = self.get_fanout();
        match self.search_key(key) {
            Some((key_exist, node_id_vec)) => {
                if !key_exist {
                    // key is a new key
                    let mut node_id_vec = node_id_vec;
                    let leaf_id = node_id_vec.pop().unwrap().node_id;
                    let mut leaf = self.load_node(leaf_id).unwrap().to_leaf().unwrap();
                    if leaf.get_n() < fanout {
                        // the leaf has capacity to insert the new data, directly insert the key-value pair to the leaf node
                        leaf = insert_in_leaf_node(leaf, key, value, fanout).unwrap();
                        // compute the digest of the leaf
                        let mut h = leaf.to_digest();
                        // store the leaf using (h, leaf) to the index storage
                        self.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));
                        // update the upper parent nodes of the leaf node
                        while node_id_vec.len() != 0 {
                            // pop the next node_id_obj
                            let node_id_obj = node_id_vec.pop().unwrap();
                            // retrieve the child index and the node_id
                            let (index, node_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                            // load the node using the node_id
                            let mut node = self.load_node(node_id).unwrap().to_internal().unwrap();
                            // update the child hash using h
                            node.child_hashes[index.unwrap()] = h;
                            // update h after the node's update
                            h = node.to_digest();
                            // store the updated node to the storage
                            self.store_node(node_id, BPlusTreeNode::from_internal(node));
                        }
                    } else {
                        // the leaf does not have enough capacity, and should be split (maybe cause the parent nodes' splits)
                        // first create a temporary key-value pairs
                        let mut temp_key_values = leaf.key_values.clone();
                        // get the inserting index of the searched key
                        let index = leaf.search_insert_idx(key);
                        // insert the key-value pair to the temporary vector
                        temp_key_values.insert(index, (key, value));
                        // compute the split position
                        let split_pos = (self.get_fanout() + 1) / 2;
                        // split the vector
                        let (temp_key_values_1, temp_key_values_2) = temp_key_values.split_at(split_pos);
                        // assigne the left split to the leaf
                        leaf.key_values = temp_key_values_1.to_vec();
                        let cur_leaf_next = leaf.next;
                        // create a new id of the right split leaf
                        let leaf_2_id = self.new_counter();
                        // link the next pointer to the right split leaf
                        leaf.next = leaf_2_id;
                        // store the left split leaf to the storage
                        self.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));

                        // split key: kprime, which is the right split leaf's first key
                        let kprime = temp_key_values_2[0].0;
                        // create right-split leaf
                        let mut leaf2 = BPlusTreeLeafNode::new(temp_key_values_2.to_vec());
                        // update the next pointer
                        leaf2.next = cur_leaf_next;
                        // store the right-split leaf
                        self.store_node(leaf_2_id, BPlusTreeNode::from_leaf(leaf2));
                        // recursively update (insert) to the parent node
                        self.insert_in_parent_node(leaf_id, leaf_2_id, kprime, node_id_vec);
                    }
                } else {
                    // key is an old key
                    let mut node_id_vec = node_id_vec;
                    let leaf_id_obj = node_id_vec.pop().unwrap();
                    let (index, leaf_id) = (leaf_id_obj.child_idx, leaf_id_obj.node_id);
                    // load the leaf node
                    let mut leaf = self.load_node(leaf_id).unwrap().to_leaf().unwrap();
                    // directly update the value at index in the leaf node
                    leaf.key_values[index.unwrap()] = (key, value);
                    // compute the leaf's digest
                    let mut h = leaf.to_digest();
                    // store the leaf node to the storage
                    self.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));
                    // iteratively update the parent nodes (updating the child hash values)
                    while node_id_vec.len() != 0 {
                        // pop the next node_id_obj
                        let node_id_obj = node_id_vec.pop().unwrap();
                        // retrieve the child index and the node_id
                        let (index, node_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                        // load the node using the node_id
                        let mut node = self.load_node(node_id).unwrap().to_internal().unwrap();
                        // update the child hash using h
                        node.child_hashes[index.unwrap()] = h;
                        // update h after the node's update
                        h = node.to_digest();
                        // store the updated node to the storage
                        self.store_node(node_id, BPlusTreeNode::from_internal(node));
                    }
                }
            },
            None => {
                // create a new node_id
                let node_id = self.new_counter();
                // create a key_value pair vector using the (key, value)
                let key_values = vec![(key, value)];
                // create a new leaf node using the vector
                let leaf = BPlusTreeLeafNode::new(key_values);
                let node = BPlusTreeNode::from_leaf(leaf);
                // store the node to the storage
                self.store_node(node_id, node);
                // update the root id
                self.set_root_id(node_id);
            }
        }
    }

    fn insert_in_parent_node(&mut self, node1_id: u32, node2_id: u32, kprime: K, node_id_vec: Vec<SearchNodeId>) {
        if node1_id == self.get_root_id() {
            // node1 is root, should create a new root containing the two child nodes, node1_id and node2_id
            // create a new node_id
            let node_id = self.new_counter();
            // create keys and childs of the new root node
            let keys = vec![kprime];
            let childs = vec![node1_id, node2_id];
            let node_1 = self.load_node(node1_id).unwrap();
            let node_2 = self.load_node(node2_id).unwrap();
            // compute the child hash values of the new root node
            let child_hashes = vec![node_1.to_digest(), node_2.to_digest()];
            // create the new root node
            let root_node = BPlusTreeInternalNode::new(keys, childs, child_hashes);
            // update the root id
            self.set_root_id(node_id);
            // store the new root node to the storage
            self.store_node(node_id, BPlusTreeNode::from_internal(root_node));
        } else {
            // the left node node1 is not the root
            let node_1 = self.load_node(node1_id).unwrap();
            let node_2 = self.load_node(node2_id).unwrap();
            // compute the two nodes' digests
            let node1_h = node_1.to_digest();
            let node2_h = node_2.to_digest();
            let mut node_id_vec = node_id_vec;
            let par_node_id = node_id_vec.pop().unwrap().node_id;
            // retrieve the direct parent node
            let mut par_node = self.load_node(par_node_id).unwrap().to_internal().unwrap();
            if par_node.get_n() < self.get_fanout() as u32 {
                // the parent node has enough space for the update
                // search the index of the kprime in the par_node
                let index = par_node.search_key_idx(kprime);
                // insert kprime to the keys vector
                par_node.keys.insert(index, kprime);
                // insert node2_id to the par node at index+1
                par_node.childs.insert(index+1, node2_id);
                // update the child hash at index and index+1
                par_node.child_hashes[index] = node1_h;
                par_node.child_hashes.insert(index+1, node2_h);
                // compute the hash of the par_node
                let mut h = par_node.to_digest();
                // store the parent node to the storage
                self.store_node(par_node_id, BPlusTreeNode::from_internal(par_node));
                // iteratively update the parent nodes' hash values
                while node_id_vec.len() > 0 {
                    // pop the next node_id_obj
                    let node_id_obj = node_id_vec.pop().unwrap();
                    // retrieve the child index and the node_id
                    let (index, par_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                    let mut node = self.load_node(par_id).unwrap().to_internal().unwrap();
                    // update the child hash at position index
                    node.child_hashes[index.unwrap()] = h;
                    // update the node hash
                    h = node.to_digest();
                    // store the updated node to the storage
                    self.store_node(par_id, BPlusTreeNode::from_internal(node));
                }
            } else {
                // parent node does not have enough space for the update
                // create temporary keys, childs, and child_hashes
                let mut temp_keys = par_node.keys.clone();
                let mut temp_childs = par_node.childs.clone();
                let mut temp_child_hashes = par_node.child_hashes.clone();
                // search the index of kprime in the par_node
                let index = par_node.search_key_idx(kprime);
                // insert kprime, node2_id, node2_h to the temp_keys, temp_childs, temp_hashes
                temp_keys.insert(index, kprime);
                temp_childs.insert(index+1, node2_id);
                // update the child hash at position index
                temp_child_hashes[index] = node1_h;
                temp_child_hashes.insert(index+1, node2_h);

                // compute the split position
                let split_pos = (self.get_fanout() + 1) / 2;
                // split the keys, childs, and child_hashes
                let (key_1, key_2) = temp_keys.split_at(split_pos);
                let (child_1, child_2) = temp_childs.split_at(split_pos+1);
                let (child_hash_1, child_hash_2) = temp_child_hashes.split_at(split_pos+1);

                par_node.keys = key_1.to_vec();
                par_node.childs = child_1.to_vec();
                par_node.child_hashes = child_hash_1.to_vec();
                // store the updated par_node
                self.store_node(par_node_id, BPlusTreeNode::from_internal(par_node));

                let mut key_2 = key_2.to_vec();
                // get the pushed-up key, which is the first key of the right-split internal node
                let kpprime = key_2.remove(0);
                // create a new node id
                let par_node2_id = self.new_counter();
                // create the right-split intenral node
                let par_node2 = BPlusTreeInternalNode::new(key_2, child_2.to_vec(), child_hash_2.to_vec());
                // store the right-split intenral node to the storage
                self.store_node(par_node2_id, BPlusTreeNode::from_internal(par_node2));
                // recursively update the parent internal nodes using the pushed-up key and the newly created right-split internal node
                self.insert_in_parent_node(par_node_id, par_node2_id, kpprime, node_id_vec);
            }
        }
    }
    // persist the temporary information in the context to the storage
    pub fn persist(&mut self) {
        self.tree.set_root_id(self.root_id);
        self.tree.set_counter(self.counter);
        self.tree.set_key_num(self.key_num);
        self.tree.store_nodes_batch(&self.nodes);
    }

    // a helper function
    pub fn change_to_raw_map(&mut self) -> (BTreeMap<Vec<u8>, Vec<u8>>, BTreeMap<Vec<u8>, Vec<u8>>, H256){
        let mut meta_map = BTreeMap::<Vec<u8>, Vec<u8>>::new();
        let mut node_map = BTreeMap::<Vec<u8>, Vec<u8>>::new();
        let key_id = self.tree.get_storage_id();
        for elem in &self.nodes {
            let node_id = elem.0;
            let node = elem.1;
            let mut node_load_key = key_id.to_be_bytes().to_vec();
            node_load_key.extend(&node_id.to_be_bytes());
            let node_byte = bincode::serialize(&node).unwrap();
            node_map.insert(node_load_key, node_byte);
        }

        let root_load_key = T::get_root_storage_key(key_id);
        let root_bytes = self.root_id.to_be_bytes().to_vec();
        meta_map.insert(root_load_key, root_bytes);

        let counter_load_key = T::get_counter_storage_key(key_id);
        let counter_bytes = self.counter.to_be_bytes().to_vec();
        meta_map.insert(counter_load_key, counter_bytes);

        let key_num_load_key = T::get_key_num_storage_key(key_id);
        let key_num_bytes = self.key_num.to_be_bytes().to_vec();
        meta_map.insert(key_num_load_key, key_num_bytes);

        let cur_root_h = self.nodes.get(&self.root_id).unwrap().to_digest();
        return (meta_map, node_map, cur_root_h);
    }
}

/* The following codes are related to the single insertion to the MB-Tree
 */
// given the node_id and the search key, get the node ids of the search path
pub fn get_search_path_ids<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>, node_id: u32, key: K) -> Option<Vec<SearchNodeId>> {
    let mut cur_id = node_id;
    let curnode = tree.load_node(node_id);
    match curnode {
        Some(node) => {
            let mut node = node;
            let mut node_ids = Vec::new();
            // iterative until the node is a leaf node
            while !node.is_leaf() {
                let internal_node = node.get_internal().unwrap();
                let index = internal_node.search_key_idx(key);
                // node id vec consists of Some(index) of the child node, and the current node id
                let search_node_id = SearchNodeId {
                    node_id: cur_id,
                    child_idx: Some(index),
                };
                node_ids.push(search_node_id);
                // update cur_id to the child node's id
                cur_id = internal_node.childs[index];
                // update the node to the child node
                node = tree.load_node(cur_id).unwrap();
            }
            let search_node_id = SearchNodeId {
                node_id: cur_id,
                child_idx: None,
            };
            node_ids.push(search_node_id);
            return Some(node_ids);
        },
        None => {
            return None; // no node exist under node_id
        }
    }
}

// given a search key, return None if the root does not exist; return Some(key_exist, leaf_node_id) otherwise
pub fn search_key<K: Num, V: Value>(tree: &impl BPlusTreeNodeIO<K, V>, key: K) -> Option<(bool, Vec<SearchNodeId>)> {
    // try to get the search path from the root_id
    let r = get_search_path_ids(tree, tree.get_root_id(), key);
    match r {
        Some(search_path) => {
            let leaf_node_id = search_path.last().unwrap().node_id;
            let leaf = tree.load_node(leaf_node_id).unwrap().to_leaf().unwrap();
            for (i, k_v) in leaf.key_values.iter().enumerate() {
                if key == k_v.0 {
                    // find the key, then update the index of the last node and return true
                    let mut node_ids = search_path;
                    let last = node_ids.last_mut().unwrap();
                    last.child_idx = Some(i);
                    return Some((true, node_ids));
                }
            }
            // does not find the key, then return false
            return Some((false, search_path));
        },
        None => {
            return None;
        }
    }
}

pub fn insert<K: Num, V: Value>(tree: &mut impl BPlusTreeNodeIO<K, V>, key: K, value: V) {
    // println!("key: {:?}, value: {:?}", key, value);
    // add 1 to the number of data
    tree.increment_key_num();
    let fanout = tree.get_fanout();
    match search_key(tree, key) {
        Some((key_exist, node_id_vec)) => {
            if !key_exist {
                // key is a new key
                let mut node_id_vec = node_id_vec;
                let leaf_id = node_id_vec.pop().unwrap().node_id;
                let mut leaf = tree.load_node(leaf_id).unwrap().to_leaf().unwrap();
                if leaf.get_n() < fanout {
                    // the leaf has capacity to insert the new data, directly insert the key-value pair to the leaf node
                    leaf = insert_in_leaf_node(leaf, key, value, fanout).unwrap();
                    // compute the digest of the leaf
                    let mut h = leaf.to_digest();
                    // store the leaf using (h, leaf) to the index storage
                    tree.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));
                    // update the upper parent nodes of the leaf node
                    while node_id_vec.len() != 0 {
                        // pop the next node_id_obj
                        let node_id_obj = node_id_vec.pop().unwrap();
                        // retrieve the child index and the node_id
                        let (index, node_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                        // load the node using the node_id
                        let mut node = tree.load_node(node_id).unwrap().to_internal().unwrap();
                        // update the child hash using h
                        node.child_hashes[index.unwrap()] = h;
                        // update h after the node's update
                        h = node.to_digest();
                        // store the updated node to the storage
                        tree.store_node(node_id, BPlusTreeNode::from_internal(node));
                    }
                } else {
                    // the leaf does not have enough capacity, and should be split (maybe cause the parent nodes' splits)
                    // first create a temporary key-value pairs
                    let mut temp_key_values = leaf.key_values.clone();
                    // get the inserting index of the searched key
                    let index = leaf.search_insert_idx(key);
                    // insert the key-value pair to the temporary vector
                    temp_key_values.insert(index, (key, value));
                    // compute the split position
                    let split_pos = (tree.get_fanout() + 1) / 2;
                    // split the vector
                    let (temp_key_values_1, temp_key_values_2) = temp_key_values.split_at(split_pos);
                    // assigne the left split to the leaf
                    leaf.key_values = temp_key_values_1.to_vec();
                    let cur_leaf_next = leaf.next;
                    // create a new id of the right split leaf
                    let leaf_2_id = tree.new_counter();
                    // link the next pointer to the right split leaf
                    leaf.next = leaf_2_id;
                    // store the left split leaf to the storage
                    tree.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));

                    // split key: kprime, which is the right split leaf's first key
                    let kprime = temp_key_values_2[0].0;
                    // create right-split leaf
                    let mut leaf2 = BPlusTreeLeafNode::new(temp_key_values_2.to_vec());
                    // update the next pointer
                    leaf2.next = cur_leaf_next;
                    // store the right-split leaf
                    tree.store_node(leaf_2_id, BPlusTreeNode::from_leaf(leaf2));
                    // recursively update (insert) to the parent node
                    insert_in_parent_node(tree, leaf_id, leaf_2_id, kprime, node_id_vec);
                }
            } else {
                // key is an old key
                let mut node_id_vec = node_id_vec;
                let leaf_id_obj = node_id_vec.pop().unwrap();
                let (index, leaf_id) = (leaf_id_obj.child_idx, leaf_id_obj.node_id);
                // load the leaf node
                let mut leaf = tree.load_node(leaf_id).unwrap().to_leaf().unwrap();
                // directly update the value at index in the leaf node
                leaf.key_values[index.unwrap()] = (key, value);
                // compute the leaf's digest
                let mut h = leaf.to_digest();
                // store the leaf node to the storage
                tree.store_node(leaf_id, BPlusTreeNode::from_leaf(leaf));
                // iteratively update the parent nodes (updating the child hash values)
                while node_id_vec.len() != 0 {
                    // pop the next node_id_obj
                    let node_id_obj = node_id_vec.pop().unwrap();
                    // retrieve the child index and the node_id
                    let (index, node_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                    // load the node using the node_id
                    let mut node = tree.load_node(node_id).unwrap().to_internal().unwrap();
                    // update the child hash using h
                    node.child_hashes[index.unwrap()] = h;
                    // update h after the node's update
                    h = node.to_digest();
                    // store the updated node to the storage
                    tree.store_node(node_id, BPlusTreeNode::from_internal(node));
                }
            }
        },
        None => {
            // create a new node_id
            let node_id = tree.new_counter();
            // create a key_value pair vector using the (key, value)
            let key_values = vec![(key, value)];
            // create a new leaf node using the vector
            let leaf = BPlusTreeLeafNode::new(key_values);
            let node = BPlusTreeNode::from_leaf(leaf);
            // store the node to the storage
            tree.store_node(node_id, node);
            // update the root id
            tree.set_root_id(node_id);
        }
    }
}

fn insert_in_parent_node<K: Num, V: Value>(tree: &mut impl BPlusTreeNodeIO<K, V>, node1_id: u32, node2_id: u32, kprime: K, node_id_vec: Vec<SearchNodeId>) {
    if node1_id == tree.get_root_id() {
        // node1 is root, should create a new root containing the two child nodes, node1_id and node2_id
        // create a new node_id
        let node_id = tree.new_counter();
        // create keys and childs of the new root node
        let keys = vec![kprime];
        let childs = vec![node1_id, node2_id];
        let node_1 = tree.load_node(node1_id).unwrap();
        let node_2 = tree.load_node(node2_id).unwrap();
        // compute the child hash values of the new root node
        let child_hashes = vec![node_1.to_digest(), node_2.to_digest()];
        // create the new root node
        let root_node = BPlusTreeInternalNode::new(keys, childs, child_hashes);
        // update the root id
        tree.set_root_id(node_id);
        // store the new root node to the storage
        tree.store_node(node_id, BPlusTreeNode::from_internal(root_node));
    } else {
        // the left node node1 is not the root
        let node_1 = tree.load_node(node1_id).unwrap();
        let node_2 = tree.load_node(node2_id).unwrap();
        // compute the two nodes' digests
        let node1_h = node_1.to_digest();
        let node2_h = node_2.to_digest();
        let mut node_id_vec = node_id_vec;
        let par_node_id = node_id_vec.pop().unwrap().node_id;
        // retrieve the direct parent node
        let mut par_node = tree.load_node(par_node_id).unwrap().to_internal().unwrap();
        if par_node.get_n() < tree.get_fanout() as u32 {
            // the parent node has enough space for the update
            // search the index of the kprime in the par_node
            let index = par_node.search_key_idx(kprime);
            // insert kprime to the keys vector
            par_node.keys.insert(index, kprime);
            // insert node2_id to the par node at index+1
            par_node.childs.insert(index+1, node2_id);
            // update the child hash at index and index+1
            par_node.child_hashes[index] = node1_h;
            par_node.child_hashes.insert(index+1, node2_h);
            // compute the hash of the par_node
            let mut h = par_node.to_digest();
            // store the parent node to the storage
            tree.store_node(par_node_id, BPlusTreeNode::from_internal(par_node));
            // iteratively update the parent nodes' hash values
            while node_id_vec.len() > 0 {
                // pop the next node_id_obj
                let node_id_obj = node_id_vec.pop().unwrap();
                // retrieve the child index and the node_id
                let (index, par_id) = (node_id_obj.child_idx, node_id_obj.node_id);
                let mut node = tree.load_node(par_id).unwrap().to_internal().unwrap();
                // update the child hash at position index
                node.child_hashes[index.unwrap()] = h;
                // update the node hash
                h = node.to_digest();
                // store the updated node to the storage
                tree.store_node(par_id, BPlusTreeNode::from_internal(node));
            }
        } else {
            // parent node does not have enough space for the update
            // create temporary keys, childs, and child_hashes
            let mut temp_keys = par_node.keys.clone();
            let mut temp_childs = par_node.childs.clone();
            let mut temp_child_hashes = par_node.child_hashes.clone();
            // search the index of kprime in the par_node
            let index = par_node.search_key_idx(kprime);
            // insert kprime, node2_id, node2_h to the temp_keys, temp_childs, temp_hashes
            temp_keys.insert(index, kprime);
            temp_childs.insert(index+1, node2_id);
            // update the child hash at position index
            temp_child_hashes[index] = node1_h;
            temp_child_hashes.insert(index+1, node2_h);

            // compute the split position
            let split_pos = (tree.get_fanout() + 1) / 2;
            // split the keys, childs, and child_hashes
            let (key_1, key_2) = temp_keys.split_at(split_pos);
            let (child_1, child_2) = temp_childs.split_at(split_pos+1);
            let (child_hash_1, child_hash_2) = temp_child_hashes.split_at(split_pos+1);

            par_node.keys = key_1.to_vec();
            par_node.childs = child_1.to_vec();
            par_node.child_hashes = child_hash_1.to_vec();
            // store the updated par_node
            tree.store_node(par_node_id, BPlusTreeNode::from_internal(par_node));

            let mut key_2 = key_2.to_vec();
            // get the pushed-up key, which is the first key of the right-split internal node
            let kpprime = key_2.remove(0);
            // create a new node id
            let par_node2_id = tree.new_counter();
            // create the right-split intenral node
            let par_node2 = BPlusTreeInternalNode::new(key_2, child_2.to_vec(), child_hash_2.to_vec());
            // store the right-split intenral node to the storage
            tree.store_node(par_node2_id, BPlusTreeNode::from_internal(par_node2));
            // recursively update the parent internal nodes using the pushed-up key and the newly created right-split internal node
            insert_in_parent_node(tree, par_node_id, par_node2_id, kpprime, node_id_vec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Serialize, Deserialize, Debug)]
    pub struct IKey(u32);
    impl Digestible for IKey {
        fn to_digest(&self) -> H256 {
            H256::from_low_u64_be(self.0 as u64)
        }
    }
    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, Serialize, Deserialize, Debug)]
    pub struct IValue(u32);
    impl Digestible for IValue {
        fn to_digest(&self) -> H256 {
            H256::from_low_u64_be(self.0 as u64)
        }
    }

    #[derive(Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
    pub struct InMemoryMBTree {
        pub root: u32,
        pub counter: u32,
        pub key_num: u32,
        pub nodes: BTreeMap<u32, BPlusTreeNode<IKey, IValue>>,
        pub fanout: usize,
    }

    impl InMemoryMBTree {
        pub fn new(fanout: usize) -> Self {
            Self { root: 0, counter: 0, key_num: 0, nodes: BTreeMap::new(), fanout: fanout }
        }
    }

    impl BPlusTreeNodeIO<IKey, IValue> for InMemoryMBTree {
        fn load_node(&self, node_id: u32) -> Option<BPlusTreeNode<IKey, IValue>> {
            match self.nodes.get(&node_id) {
                Some(n) => Some(n.clone()),
                None => None,
            }
        }

        fn store_node(&mut self, node_id: u32, node: BPlusTreeNode<IKey, IValue>) {
            self.nodes.insert(node_id, node);
        }

        fn store_nodes_batch(&mut self, nodes: &BTreeMap::<u32, BPlusTreeNode<IKey, IValue>>) {
            self.nodes.extend(nodes.clone());
        }

        fn new_counter(&mut self) -> u32 {
            self.counter += 1;
            self.counter
        }

        fn set_counter(&mut self, counter: u32) {
            self.counter = counter
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
            if self.key_num == 0 {
                return H256::default();
            } else {
                self.nodes.get(&self.root).unwrap().to_digest()
            }
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

        fn load_all_key_values(&self) -> Vec<(IKey, IValue)> {
            let mut values = Vec::<(IKey, IValue)>::new();
            if self.key_num != 0 {
                // get the left most leaf node's id
                let mut cur_leaf_id = get_left_most_leaf_id(self);
                // iteratively scan the leaf nodes from left to right until the leaf's next pointer is 0
                while cur_leaf_id != 0 {
                    let leaf = self.load_node(cur_leaf_id).unwrap().to_leaf().unwrap();
                    for i in 0..leaf.get_n() {
                        let (key, value) = leaf.key_values[i];
                        values.push((key, value));
                    }
                    cur_leaf_id = leaf.next;
                }
            }
            return values;
        }

        fn get_storage_id(&self,) -> u64 {
            0
        }

        fn get_root_storage_key(_: u64) -> Vec<u8> {
            vec![]
        }

        fn get_counter_storage_key(_: u64) -> Vec<u8> {
            vec![]
        }

        fn get_key_num_storage_key(_: u64) -> Vec<u8> {
            vec![]
        }

        fn set_fanout(&mut self, fanout: usize) {
            self.fanout = fanout;
        }

        fn get_fanout(&self) -> usize {
            self.fanout
        }
    }

    #[test]
    fn test_in_memory_mbtree() {
        let fanout = 4;
        let n = 1234;
        let mut tree = InMemoryMBTree::new(fanout);
        for i in 1..=n {
            let key = IKey(i * 2);
            let value = IValue(i * 2);
            insert(&mut tree, key, value);
        }
        println!("complete insertion");
        /* let mut mbtree_write_context = WriteContext::new(&mut tree);
        for i in 1..=n {
            let key = IKey(i * 2);
            let value = IValue(i * 2);
            mbtree_write_context.insert(key, value);
        }
        mbtree_write_context.persist(); */

        // let lb = IKey(3);
        // let ub = IKey(21);
        // let (r, p) = get_range_proof(&tree, lb, ub);
        // println!("r: {:?}", r);
        // let h = reconstruct_range_proof(lb, ub, &r, &p);
        // assert_eq!(h, tree.get_root_hash());
        for i in 1..=n {
            for j in i..=n {
                let lb = IKey(i);
                let rb = IKey(j);
                let (r, p) = get_range_proof(&tree, lb, rb);
                let h = reconstruct_range_proof(lb, rb, &r, &p);
                assert_eq!(h, tree.get_root_hash());
            }
        }
    }

    #[test]
    fn test_empty_tree() {
        let fanout = 4;
        let tree = InMemoryMBTree::new(fanout);
        let lb = IKey(3);
        let rb = IKey(4);
        let (r, p) = get_range_proof(&tree, lb, rb);
        let h = reconstruct_range_proof(lb, rb, &r, &p);
        assert_eq!(h, tree.get_root_hash());
    }
}