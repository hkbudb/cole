use utils::types::{Num, Value, Digestible};
use utils::{H256, types::bytes_hash};
use serde::{Serialize, Deserialize};

/* Leaf node of a B+-Tree
    key_values: store the key-value pairs in the leaf node
    next: the pointer to the next leaf node
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BPlusTreeLeafNode<K: Num, V: Value> {
    pub key_values: Vec<(K, V)>,
    pub next: u32,
}

impl<K: Num, V: Value> Digestible for BPlusTreeLeafNode<K, V> {
    fn to_digest(&self) -> H256 {
        // digest of a leaf node is H(H(k_0)||H(v_0)||H(k_1)||H(v_1)|| ... H(k_f)||H(v_f))
        let mut bytes = vec![];
        for (k, v) in &self.key_values {
            let k_h = k.to_digest();
            let v_h = v.to_digest();
            bytes.extend(k_h.as_bytes());
            bytes.extend(v_h.as_bytes());
        }
        bytes_hash(&bytes)
    }
}

impl<K: Num, V: Value> BPlusTreeLeafNode<K, V> {
    // create a new leaf node with key-value pairs
    pub fn new(key_values: Vec<(K, V)>) -> Self {
        Self {
            key_values,
            next: 0,
        }
    }
    // get number of keys
    pub fn get_n(&self) -> usize {
        self.key_values.len()
    }

    pub fn search_insert_idx(&self, key: K) -> usize {
        let mut i: usize = 0;
        while i < self.get_n() {
            if key < self.key_values[i].0 {
                break;
            }
            i += 1;
        }
        return i;
    }

    pub fn search_prove_idx(&self, key: K) -> usize {
        let mut i: usize = 0;
        let n = self.get_n() as i32;
        while (i as i32) < n - 1 {
            // if key is smaller or equal to the first key in the leaf node, index is 0
            if key <= self.key_values[0].0 {
                return 0;
            }
            // if key[i] <= key < key[i+1], index is i 
            else if key >= self.key_values[i].0 && key < self.key_values[i+1].0 {
                return i;
            } else {
                // otherwise, increment i until i == n - 1, which is the last index of the leaf node
                i += 1;
            }
        }
        return i;
    }

    pub fn search_prove_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        let mut i: usize = 0;
        let n = self.get_n() as i32;
        while (i as i32) < n - 1 {
            // if key is smaller or equal to the first key in the leaf node, index is 0
            if lb <= self.key_values[0].0 {
                break;
            }
            // if key[i] <= key < key[i+1], index is i 
            else if lb >= self.key_values[i].0 && lb < self.key_values[i+1].0 {
                break;
            } else {
                // otherwise, increment i until i == n - 1, which is the last index of the leaf node
                i += 1;
            }
        }

        let mut j: usize = 0;
        while (j as i32) < n - 1 {
            // if key is smaller or equal to the first key in the leaf node, index is 0
            if ub <= self.key_values[0].0 {
                break;
            }
            // if key[j] <= key < key[j+1], index is j 
            else if ub >= self.key_values[j].0 && ub < self.key_values[j+1].0 {
                break;
            } else {
                // otherwise, increment j until j == n - 1, which is the last index of the leaf node
                j += 1;
            }
        }
        return (i, j);
    }
}

/* Internal node of a B+-Tree
    keys: the direction keys to the child nodes
    childs: the pointers of the child nodes
    child_hashes: the hash values of the child nodes
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BPlusTreeInternalNode<K: Num> {
    pub keys: Vec<K>,
    pub childs: Vec<u32>,
    pub child_hashes: Vec<H256>,
}

impl<K: Num> Digestible for BPlusTreeInternalNode<K> {
    // digest of an internal node is computed from H(H(k_0) || H(k_1) || ... || H(k_f) || H(h_0) || H(h_1) || ... H(h_{f+1})), assume that there are f+1 child nodes
    fn to_digest(&self) -> H256 {
        let mut bytes = vec![];
        for k in &self.keys {
            let h = k.to_digest();
            bytes.extend(h.as_bytes());
        }

        for h in &self.child_hashes {
            bytes.extend(h.as_bytes());
        }
        bytes_hash(&bytes)
    }
}

impl<K: Num> BPlusTreeInternalNode<K> {
    // create an internal node
    pub fn new(keys: Vec<K>, childs: Vec<u32>, child_hashes: Vec<H256>) -> Self {
        Self {
            keys,
            childs,
            child_hashes,
        }
    }
    // given an index, get the node id of the child node
    pub fn get_child_id(&self, idx: usize) -> u32 {
        *self.childs.get(idx).unwrap()
    }
    // given an index, get the hash value of the child node
    pub fn get_child_hash(&self, idx: usize) -> H256 {
        *self.child_hashes.get(idx).unwrap()
    }
    // get the number of keys in the internal node
    pub fn get_n(&self) -> u32 {
        self.keys.len() as u32
    }

    pub fn search_key_idx(&self, key: K) -> usize {
        let mut i: usize = 0;
        while (i as u32) < self.get_n() {
            if key < self.keys[i] {
                break;
            }
            i += 1;
        }
        return i;
    }

    // search the index range of the searched key range in the internal node
    pub fn search_key_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        let mut i: usize = 0;
        let n = self.get_n();
        while (i as u32) < n {
            if lb < self.keys[i] {
                break;
            }
            i += 1;
        }

        let mut j: usize = 0;
        let n = self.get_n();
        while (j as u32) < n {
            if ub < self.keys[j] {
                break;
            }
            j += 1;
        }

        return (i, j);
    }
}

/* The Enumerator of the MB-Tree node: either a leaf node or an internal node
 */
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum BPlusTreeNode<K: Num, V: Value> {
    Leaf(BPlusTreeLeafNode<K, V>),
    NonLeaf(BPlusTreeInternalNode<K>),
}

impl<K: Num, V: Value> Digestible for BPlusTreeNode<K, V> {
    fn to_digest(&self) -> H256 {
        match self {
            BPlusTreeNode::NonLeaf(n) => n.to_digest(),
            BPlusTreeNode::Leaf(n) => n.to_digest(),
        }
    }
}

// try to transform a node to a leaf node, if the node is not a leaf node, return None
impl<K: Num, V: Value> Into<Option<BPlusTreeLeafNode<K, V>>> for BPlusTreeNode<K, V> {
    fn into(self) -> Option<BPlusTreeLeafNode<K, V>> {
        match self {
            BPlusTreeNode::Leaf(n) => Some(n),
            BPlusTreeNode::NonLeaf(_) => None,
        }
    }
}

impl<K: Num, V: Value> BPlusTreeNode<K, V> {
    pub fn get_keys(&self) -> Vec<K> {
        match self {
            BPlusTreeNode::Leaf(n) => {
                let keys: Vec<K> = n.key_values.iter().map(|(k, _)| *k).collect();
                keys
            },
            BPlusTreeNode::NonLeaf(n) => n.keys.clone(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            BPlusTreeNode::Leaf(_) => true,
            BPlusTreeNode::NonLeaf(_) => false,
        }
    }

    pub fn to_leaf(self) -> Option<BPlusTreeLeafNode<K, V>> {
        match self {
            BPlusTreeNode::Leaf(n) => Some(n),
            BPlusTreeNode::NonLeaf(_) => None,
        }
    }

    pub fn get_leaf(&self) -> Option<&BPlusTreeLeafNode<K, V>> {
        match self {
            BPlusTreeNode::Leaf(n) => Some(n),
            BPlusTreeNode::NonLeaf(_) => None,
        }
    }

    pub fn get_internal(&self) -> Option<&BPlusTreeInternalNode<K>> {
        match self {
            BPlusTreeNode::Leaf(_) => None,
            BPlusTreeNode::NonLeaf(n) => Some(n),
        }
    }

    pub fn to_internal(self) -> Option<BPlusTreeInternalNode<K>> {
        match self {
            BPlusTreeNode::Leaf(_) => None,
            BPlusTreeNode::NonLeaf(n) => Some(n),
        }
    }

    pub fn from_leaf(leaf: BPlusTreeLeafNode<K, V>) -> Self {
        Self::Leaf(leaf)
    }

    pub fn from_internal(non_leaf: BPlusTreeInternalNode<K>) -> Self {
        Self::NonLeaf(non_leaf)
    }

    pub fn search_prove_idx(&self, key: K) -> usize {
        match &self {
            BPlusTreeNode::Leaf(node) => node.search_prove_idx(key),
            BPlusTreeNode::NonLeaf(node) => node.search_key_idx(key),
        }
    }

    pub fn search_prove_idx_range(&self, lb: K, ub: K) -> (usize, usize) {
        match &self {
            BPlusTreeNode::Leaf(node) => node.search_prove_idx_range(lb, ub),
            BPlusTreeNode::NonLeaf(node) => node.search_key_idx_range(lb, ub),
        }
    }
    
    // get the child id given the index
    pub fn get_internal_child(&self, index: usize) -> Option<u32> {
        match &self {
            BPlusTreeNode::Leaf(_) => None,
            BPlusTreeNode::NonLeaf(node) => {
                return Some(node.childs[index])
            },
        }
    }
}