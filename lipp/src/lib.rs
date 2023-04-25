use kvdb_rocksdb::Database;
use rug::{Rational, Integer, Complete};
use utils::{types::{Num, Value, Digestible, BigNum, AddrKey, StateValue, bytes_hash}, H256, ROCKSDB_COL_ID};
use serde::{Serialize, Deserialize};
use growable_bitmap::GrowableBitMap;
use std::{collections::{BTreeMap, HashMap, HashSet}, cmp::min};

/* Item records a pair of key-value
 */
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Item<K: Num + BigNum, V: Value> {
    pub key: K,
    pub value: V,
}

impl<K: Num + BigNum, V: Value> Item<K, V> {
    pub fn new_items(n: u32) -> Vec<Self> {
        return vec![Item::default(); n as usize];
    }
}
/* Linear model consists of the slope and intercept, which are both big rational numbers to keep the precision.
 */
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct LinearModel {
    slope: Rational,
    intercept: Rational,
}

impl LinearModel {
    pub fn new(a: Rational, b: Rational) -> Self {
        Self { slope: a, intercept: b }
    }

    pub fn predict(&self, key: Integer) -> usize {
        let pos = (key * &self.slope).complete() + &self.intercept;
        return pos.to_f64().floor() as usize;
    }

    pub fn predict_double(&self, key: &Integer) -> f64 {
        let pos = (key * &self.slope).complete() + &self.intercept;
        return pos.to_f64();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node<K: Num + BigNum, V: Value> {
    pub is_two: u8, // is special node for only two keys
    pub build_size: u32, // tree size (include sub nodes) when node created
    pub size: u32, // current tree size (include sub nodes)
    pub fixed: bool, // fixed node will not trigger rebuild
    pub num_inserts: u32,
    pub num_insert_to_data: u32,
    pub num_items: u32, // size of items
    pub model: LinearModel, 
    pub items: Vec<Item<K, V>>,
    pub none_bitmap: GrowableBitMap<u8>, // 1 means None, 0 means Data or Child
    pub child_bitmap: GrowableBitMap<u8>, // 1 means Child. will always be 0 when none_bitmap is 1
    pub child_hashes: BTreeMap<usize, H256>,
}

impl<K: Num + BigNum, V: Value> Node<K, V> {
    pub fn predict_pos(&self, key: &Integer) -> usize {
        let v = self.model.predict_double(key);
        if v > (u64::MAX / 2) as f64 {
            return self.num_items as usize - 1;
        }
        if v < 0.0 {
            return 0;
        }
        let pos =  min(self.num_items - 1, v as u32) as usize;
        return pos;
    }

    pub fn new() -> Self {
        Self {
            is_two: 0,
            build_size: 0,
            size: 0,
            fixed: false,
            num_inserts: 0,
            num_insert_to_data: 0,
            num_items: 1,
            model: LinearModel::default(),
            items: Item::new_items(1),
            none_bitmap: GrowableBitMap::<u8>::new(),
            child_bitmap: GrowableBitMap::<u8>::new(),
            child_hashes: BTreeMap::new(),
        }
    }

    pub fn build_tree_none() -> Self {
        let mut none_bitmap = GrowableBitMap::<u8>::new();
        none_bitmap.set_bit(0);
        Self {
            is_two: 0,
            build_size: 0,
            size: 0,
            fixed: false,
            num_inserts: 0,
            num_insert_to_data: 0,
            num_items: 1,
            model: LinearModel::default(),
            items: Item::new_items(1),
            none_bitmap,
            child_bitmap: GrowableBitMap::<u8>::new(),
            child_hashes: BTreeMap::new(),
        }
    }

    pub fn build_tree_two(key1: K, value1: V, key2: K, value2: V) -> Self {
        let mut tkey_1 = key1;
        let mut tvalue_1 = value1;
        let mut tkey_2 = key2;
        let mut tvalue_2 = value2;
        if key1 > key2 {
            tkey_1 = key2;
            tvalue_1 = value2;
            tkey_2 = key1;
            tvalue_2 = value1;
        }

        let mut node = Node::<K, V>::new();
        node.is_two = 1;
        node.build_size = 2;
        node.size = 2;
        node.fixed = false;
        node.num_inserts = 0;
        node.num_insert_to_data = 0;
        node.num_items = 8;
        node.items = Item::new_items(node.num_items);
        for i in 0..node.num_items {
            node.none_bitmap.set_bit(i as usize);
        }

        let mid1_key = &tkey_1;
        let mid2_key = &tkey_2;

        let mid1_target = node.num_items / 3;
        let mid2_target = node.num_items * 2 / 3;

        let mid1_key_big = mid1_key.to_big_integer();
        let mid2_key_big = mid2_key.to_big_integer();
        
        node.model.slope = Rational::from_f64((mid2_target - mid1_target) as f64).unwrap()  / (&mid2_key_big - &mid1_key_big).complete();
        node.model.intercept = Rational::from_f64(mid1_target as f64).unwrap() - node.model.slope.clone() * &mid1_key_big;

        // insert key1&value1
        let pos = node.predict_pos(&mid1_key_big);
        assert!(node.none_bitmap.get_bit(pos));
        node.none_bitmap.clear_bit(pos);
        node.items[pos].key = tkey_1;
        node.items[pos].value = tvalue_1;

        // insert key2&value2
        let pos = node.predict_pos(&mid2_key_big);
        assert!(node.none_bitmap.get_bit(pos));
        node.none_bitmap.clear_bit(pos);
        node.items[pos].key = tkey_2;
        node.items[pos].value = tvalue_2;

        return node;
    }
}

impl<K: Num + BigNum, V: Value> Digestible for Node<K, V> {
    fn to_digest(&self) -> H256 {
        if self.size == 0 {
            // empty node
            return H256::default();
        } else {
            let mut bytes = vec![];
            for i in 0..self.num_items as usize {
                if self.none_bitmap.get_bit(i) == false {
                    if self.child_bitmap.get_bit(i) == true {
                        // should include child hash
                        assert!(self.child_hashes.contains_key(&i));
                        let child_h = self.child_hashes.get(&i).unwrap();
                        bytes.extend(child_h.as_bytes());
                    } else {
                        // should be a data node
                        let key = &self.items[i].key;
                        let value = self.items[i].value;
                        let key_hash = key.to_digest();
                        let value_hash = value.to_digest();
                        let mut key_value_bytes = key_hash.as_bytes().to_vec();
                        key_value_bytes.extend(value_hash.as_bytes());
                        let h = bytes_hash(&key_value_bytes);
                        bytes.extend(h.as_bytes());
                    }
                }
            }
            let final_h = bytes_hash(&bytes);
            return final_h;
        }
        /* let mut cnt = 0;
        for i in 0..self.num_items as usize {
            if self.none_bitmap.get_bit(i) == false {
                if self.child_bitmap.get_bit(i) == true {
                    // should include child hash
                    assert!(self.child_hashes.contains_key(&i));
                    let child_h = self.child_hashes.get(&i).unwrap();
                    hasher.update(child_h.as_bytes());
                } else {
                    // should be a data node
                    let key = &self.items[i].key;
                    let value = self.items[i].value;
                    let mut inner_hasher = Params::new().hash_length(32).to_state();
                    let key_hash = key.to_digest();
                    let value_hash = value.to_digest();
                    inner_hasher.update(key_hash.as_bytes());
                    inner_hasher.update(value_hash.as_bytes());
                    let data_h = inner_hasher.finalize();
                    hasher.update(data_h.as_bytes());
                }
            } else {
                cnt += 1;
            }
        }
        if cnt == self.num_items {
            // empty node
            return H256::default();
        } else {
            return H256::from_slice(hasher.finalize().as_bytes());
        } */
    }
}

pub trait NodeLoader<K: Num + BigNum, V: Value> {
    fn load_node(&self, address: H256) -> Node<K, V>;
}

#[derive(Debug, Clone)]
pub struct Apply<K: Num + BigNum, V: Value> {
    pub root: H256,
    pub nodes: HashMap<H256, Node<K, V>>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Proof<K: Num + BigNum, V: Value> {
    proof: BTreeMap<H256, Node<K, V>>,
    root: H256,
    pos: usize,
}

pub struct WriteContext<L: NodeLoader<K, V>, K: Num + BigNum, V: Value> {
    pub node_loader: L,
    pub apply: Apply<K, V>,
    pub outdated: HashSet<H256>,
}

impl<L: NodeLoader<K, V>, K: Num + BigNum, V: Value> WriteContext<L, K, V> {
    pub fn new(node_loader: L, root_address: H256) -> Self {
        Self {
            node_loader,
            apply: Apply {
                root: root_address,
                nodes: HashMap::new(),
            },
            outdated: HashSet::new(),
        }
    }

    pub fn changes(self) -> Apply<K, V> {
        self.apply
    }

    fn get_node(&self, address: H256) -> Node<K, V> {
        match self.apply.nodes.get(&address) {
            Some(n) => {
                n.clone()
            },
            None => {
                self.node_loader.load_node(address)
            }
        }
    }

    fn scan_tree(&mut self, root: H256, size: usize) -> (Vec<K>, Vec<V>) {
        let mut keys = vec![K::default(); size];
        let mut values = vec![V::default(); size];
        #[derive(Debug)]
        struct Segment {
            begin: usize,
            node_id: H256,
        }

        let mut s = Vec::<Segment>::new();
        s.push(Segment {
            begin: 0,
            node_id: root,
        });

        while !s.is_empty() {
            let last = s.pop().unwrap();
            let mut begin = last.begin;
            let node_id = last.node_id;
            self.outdated.insert(node_id);
            let node = self.get_node(node_id);
            let should_end_pos = begin + node.size as usize;
            for i in 0..node.num_items as usize {
                if node.none_bitmap.get_bit(i) == false {
                    if node.child_bitmap.get_bit(i) == false {
                        keys[begin] = node.items[i].key;
                        values[begin] = node.items[i].value;
                        begin += 1;
                    } else {
                        let child_id = node.child_hashes.get(&i).unwrap().clone();
                        s.push(Segment { begin: begin, node_id: child_id });
                        begin += self.get_node(child_id).size as usize;
                    }
                }
            }
            assert!(should_end_pos == begin);
        }
        return (keys, values);
    }

    #[allow(non_snake_case)]
    pub fn build_tree_bulk_fmcd(&mut self, keys: Vec<K>, values: Vec<V>, size: usize) -> H256 {
        assert!(size >= 2);
        if size == 2 {
            // println!("size two keys: {:?}", keys);
            let node = Node::build_tree_two(keys[0], values[0], keys[1], values[1]);
            let node_h = node.to_digest();
            self.outdated.remove(&node_h);
            self.apply.nodes.insert(node_h, node);
            return node_h;
        } else {
            let build_gap_cnt = compute_gap_count(size as usize);
            let mut node = Node::new();
            node.is_two = 0;
            node.build_size = size as u32;
            node.size = size as u32;
            node.fixed = false;
            node.num_inserts = 0;
            node.num_insert_to_data = 0;
            // FMCD method
            // Here the implementation is a little different with Algorithm 1 in our paper.
            // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
            // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
            // So we added a small number (1e-6) to U_T.
            // In fact, it has only a negligible impact of the performance.
            let L = size as usize * (build_gap_cnt + 1);
            let mut i: usize = 0;
            let mut D: usize = 1;
            assert!(D <= size as usize - 1 - D);
            let mut Ut = ((&keys[size as usize - 1 - D].to_big_integer() - &keys[D].to_big_integer()).complete() / Rational::from_f64((L - 2) as f64).unwrap()) + Rational::from_f64(1e-6).unwrap();
            // let mut Ut = (keys[size as usize - 1 - D] - keys[D]) / (L - 2) as f64 + 1e-6;
            while i < size as usize - 1 - D {
                while i + D < size as usize && (&keys[i + D].to_big_integer() - &keys[i].to_big_integer()).complete() >= Ut {
                    i += 1;
                }

                if (i + D) >= size as usize {
                    break;
                }

                D += 1;
                if D * 3 > size as usize {
                    break;
                }
                assert!(D <= size as usize - 1 - D);
                Ut = ((&keys[size as usize - 1 - D].to_big_integer() - &keys[D].to_big_integer()).complete() / Rational::from_f64((L - 2) as f64).unwrap()) + Rational::from_f64(1e-6).unwrap();
            }

            if D * 3 <= size as usize {
                // println!("fmcd");
                node.model.slope = Rational::from_f64(1.0).unwrap() / Ut;
                node.model.intercept = (Rational::from_f64(L as f64).unwrap() - node.model.slope.clone() * (&keys[size as usize - 1 - D].to_big_integer() + &keys[D].to_big_integer()).complete()) / Rational::from_f64(2.0).unwrap();
                node.num_items = L as u32;
            } else {
                // println!("fail fmcd");
                let mid1_pos = (size as u32 - 1) / 3;
                let mid2_pos = (size as u32 - 1) * 2 / 3;
                assert!(mid1_pos < mid2_pos);
                assert!(mid2_pos < size as u32 - 1);

                let mid1_key = (&keys[mid1_pos as usize].to_big_integer() + &keys[(mid1_pos+1) as usize].to_big_integer()).complete() / Rational::from_f64(2.0).unwrap();
                let mid2_key = (&keys[mid2_pos as usize].to_big_integer() + &keys[(mid2_pos+1) as usize].to_big_integer()).complete() / Rational::from_f64(2.0).unwrap();

                node.num_items = size as u32 * (build_gap_cnt + 1) as u32;
                let mid1_target = mid1_pos * (build_gap_cnt + 1) as u32 + ((build_gap_cnt + 1) / 2) as u32;
                let mid2_target = mid2_pos * (build_gap_cnt + 1) as u32 + ((build_gap_cnt + 1) / 2) as u32;

                node.model.slope = Rational::from_f64(mid2_target as f64 - mid1_target as f64).unwrap() / (&mid2_key - &mid1_key).complete();
                node.model.intercept = Rational::from_f64(mid1_target as f64).unwrap() - node.model.slope.clone() * mid1_key;
            }

            assert!(node.model.slope >= 0.0);
            if size > 1000000 {
                node.fixed = true;
            }

            node.items = Item::new_items(node.num_items);
            for i in 0..node.num_items {
                node.none_bitmap.set_bit(i as usize);
            }

            let mut item_i = node.predict_pos(&keys[0].to_big_integer());
            let mut offset = 0;
            while offset < size {
                let mut next = offset + 1;
                let mut next_i = 0;
                while next < size {
                    next_i = node.predict_pos(&keys[next as usize].to_big_integer());
                    if next_i == item_i {
                        next += 1;
                    } else {
                        break;
                    }
                }

                if next == offset + 1 {
                    node.none_bitmap.clear_bit(item_i);
                    node.items[item_i].key = keys[offset as usize];
                    node.items[item_i].value = values[offset as usize];
                } else {
                    node.none_bitmap.clear_bit(item_i);
                    node.child_bitmap.set_bit(item_i);
                    let start = offset;
                    let end = next;
                    let child_id = self.build_tree_bulk_fmcd(keys[start..end].to_vec(), values[start..end].to_vec(), end - start);
                    node.child_hashes.insert(item_i, child_id);
                }

                if next >= size {
                    break;
                } else {
                    item_i = next_i;
                    offset = next;
                }
            }
            let node_h = node.to_digest();
            self.outdated.remove(&node_h);
            self.apply.nodes.insert(node_h, node);
            return node_h;
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        // println!("insert: {}, root: {:?}", key, self.apply.root);
        let key_big_int = key.to_big_integer();
        let mut temporary_path = Vec::<(Node<K, V>, Option<usize>)>::new(); // (node, child_idx)
        let mut insert_to_data = 0;
        self.outdated.insert(self.apply.root);
        let mut node = self.get_node(self.apply.root);
        let mut update_flag = false;
        loop {
            node.size += 1;
            node.num_inserts += 1;
            let pos = node.predict_pos(&key_big_int);
            if node.none_bitmap.get_bit(pos) == true {
                node.none_bitmap.clear_bit(pos);
                node.items[pos].key = key;
                node.items[pos].value = value;
                temporary_path.push((node.clone(), None));
                break;
            }
            else if node.child_bitmap.get_bit(pos) == false {
                if node.items[pos].key == key {
                    // update case, since the key is the same
                    update_flag = true;
                    node.items[pos].value = value;
                    temporary_path.push((node.clone(), None));
                } else {
                    node.child_bitmap.set_bit(pos);
                    let item = &node.items[pos];
                    let new_node = Node::build_tree_two(key, value, item.key, item.value);
                    let new_node_h = new_node.to_digest();
                    node.child_hashes.insert(pos, new_node_h);
                    insert_to_data = 1;
                    temporary_path.push((node.clone(), Some(pos)));
                    temporary_path.push((new_node, None));
                }
                break;
            } else {
                let child_address = node.child_hashes.get(&pos).unwrap().clone();
                temporary_path.push((node.clone(), Some(pos)));
                self.outdated.insert(child_address);
                node = self.get_node(child_address);
            }
        }

        let mut new_root = H256::zero();
        for (i, (node, idx)) in temporary_path.iter_mut().rev().enumerate() {
            node.num_insert_to_data += insert_to_data;
            if i == 0 {
                if update_flag {
                    node.size -= 1;
                    node.num_inserts -= 1;
                }
                new_root = node.to_digest();
                self.outdated.remove(&new_root);
                self.apply.nodes.insert(new_root, node.clone());

            } else {
                let child_idx = idx.unwrap();
                node.child_hashes.insert(child_idx, new_root);
                if update_flag {
                    node.size -= 1;
                    node.num_inserts -= 1;
                }
                new_root = node.to_digest();
                self.outdated.remove(&new_root);
                self.apply.nodes.insert(new_root, node.clone());
            }
        }
        self.apply.root = new_root;
        // search the key and get the search path
        let mut search_path = Vec::<(Node<K, V>, Option<usize>)>::new(); // (node, child_idx)

        let mut node = self.get_node(self.apply.root);
        loop {
            let pos = node.predict_pos(&key_big_int);
            search_path.push((node.clone(), Some(pos)));
            if node.child_bitmap.get_bit(pos) == true {
                let node_id = node.child_hashes.get(&pos).unwrap().clone();
                node = self.get_node(node_id);
            } else {
                if node.none_bitmap.get_bit(pos) == true {
                    assert!(false);
                }
                if node.items[pos].key == key {
                    break;
                } else {
                    assert!(false);
                    break;
                }
            }
        }

        let mut is_rebuild = false;
        let mut max_level = search_path.len() - 1;
        for i in 0..search_path.len() {
            let (node, _) = search_path.get_mut(i).unwrap();
            let num_inserts = node.num_inserts;
            let num_insert_to_data = node.num_insert_to_data;
            let need_rebuild = node.fixed == false && node.size >= node.build_size * 4 && node.size >= 64 && num_insert_to_data * 10 >= num_inserts;
            if need_rebuild {
                is_rebuild = true;
                // println!("rebuild level {}", i);
                let esize = node.size;
                let (keys, values) = self.scan_tree(node.to_digest(), esize as usize);
                let new_node_h = self.build_tree_bulk_fmcd(keys, values, esize as usize);
                let cur_node = self.get_node(new_node_h);
                *node = cur_node;
                if i > 0 {
                    let (parent_node, idx) = search_path.get_mut(i-1).unwrap();
                    parent_node.child_hashes.insert(idx.unwrap(), new_node_h);
                    max_level = i - 1;
                } else {
                    max_level = 0;
                }
                break;
            }
        }

        if is_rebuild {
            search_path = search_path[0..=max_level].to_vec();
            let mut new_root = H256::zero();
            for (i, (node, idx)) in search_path.iter_mut().rev().enumerate() {
                if i == 0 {
                    new_root = node.to_digest();
                    self.apply.nodes.insert(new_root, node.clone());
    
                } else {
                    let child_idx = idx.unwrap();
                    node.child_hashes.insert(child_idx, new_root);
                    new_root = node.to_digest();
                    self.outdated.remove(&new_root);
                    self.apply.nodes.insert(new_root, node.clone());
                }
            }
            self.apply.root = new_root;
        }
        
        for addr in self.outdated.drain() {
            self.apply.nodes.remove(&addr);
        }
    }

    pub fn traverse(&self) {
        println!("traverse");
        let root_id = self.apply.root;
        let node = self.get_node(root_id);
        let mut s = vec![node];
        while s.len() != 0 {
            let node = s.pop().unwrap();
            println!("{:?} {:?}", node.to_digest(), node);
            for i in 0..node.num_items as usize {
                if node.none_bitmap.get_bit(i) == false {
                    if node.child_bitmap.get_bit(i) == true {
                        let child_id = node.child_hashes.get(&i).unwrap().clone();
                        let node = self.get_node(child_id);
                        s.push(node);
                    }
                }
            }
        }
    }
}

pub fn read_tree_without_proof<K: Num + BigNum, V: Value>(node_loader: &impl NodeLoader<K, V>, root_address: H256, key: K) -> Option<V> {
    // println!("key: {}", key);
    let result;
    let key_big_num = key.to_big_integer();
    let mut node = node_loader.load_node(root_address);
    loop {
        let pos = node.predict_pos(&key_big_num);
        // println!("node: {:?}, pos: {}", node, pos);
        if node.child_bitmap.get_bit(pos) == true {
            let node_id = node.child_hashes.get(&pos).unwrap().clone();
            node = node_loader.load_node(node_id);
        } else {
            if node.none_bitmap.get_bit(pos) == true {
                result = None;
                break;
            }

            if node.items[pos].key == key {
                result = Some(node.items[pos].value);
                break;
            } else {
                result = None;
                break;
            }
        }
    }
    return result;
}

pub fn read_tree<K: Num + BigNum, V: Value>(node_loader: &impl NodeLoader<K, V>, root_address: H256, key: K) -> (Option<V>, Proof<K, V>) {
    let mut p = Proof::default();
    p.root = root_address;
    let result;
    let key_big_num = key.to_big_integer();
    let mut node = node_loader.load_node(root_address);
    loop {
        p.proof.insert(node.to_digest(), node.clone());
        let pos = node.predict_pos(&key_big_num);
        // println!("node: {:?}, pos: {}", node, pos);
        if node.child_bitmap.get_bit(pos) == true {
            let node_id = node.child_hashes.get(&pos).unwrap().clone();
            node = node_loader.load_node(node_id);
        } else {
            if node.none_bitmap.get_bit(pos) == true {
                result = None;
                break;
            }

            if node.items[pos].key == key {
                p.pos = pos;
                result = Some(node.items[pos].value);
                break;
            } else {
                result = None;
                break;
            }
        }
    }
    return (result, p);
}

pub fn verify<K: Num + BigNum, V: Value>(key: K, value: Option<V>, root_h: H256, proof: &Proof<K, V>) -> bool {
    let mut error_flag = true;
    let load_root = proof.root;
    let key_big_num = key.to_big_integer();
    let mut node = proof.proof.get(&load_root).unwrap();
    loop {
        let pos = node.predict_pos(&key_big_num);
        if node.child_bitmap.get_bit(pos) == true {
            let node_id = node.child_hashes.get(&pos).unwrap();
            node = proof.proof.get(&node_id).unwrap();
        } else {
            if node.none_bitmap.get_bit(pos) == true {
                if value != None {
                    error_flag = false;
                }
                break;
            }

            if node.items[pos].key == key {
                if value.unwrap() != node.items[pos].value {
                    error_flag = false;
                }
                break;
            } else {
                if value != None {
                    error_flag = false;
                }
                break;
            }
        }
    }

    if load_root != root_h {
        error_flag = false;
    }

    return error_flag;
}

pub fn exists<K: Num + BigNum, V: Value>(node_loader: &impl NodeLoader<K, V>, root_address: H256, key: K) -> bool {
    let mut node = node_loader.load_node(root_address);
    let key_big_num = key.to_big_integer();
    loop {
        let pos = node.predict_pos(&key_big_num);
        if node.none_bitmap.get_bit(pos) == true {
            return false;
        } else if node.child_bitmap.get_bit(pos) == false {
            return node.items[pos].key == key;
        } else {
            let node_id = node.child_hashes.get(&pos).unwrap().clone();
            node = node_loader.load_node(node_id);
        }
    }
}

pub fn compute_gap_count(size: usize) -> usize{
    if size >= 1000000 {
        return 1;
    }
    if size >= 100000 {
        return 2;
    }
    return 5;
}

pub struct PersistLipp<'a> {
    pub roots: Vec<H256>,
    pub db: &'a Database,
}

impl<'a> Drop for PersistLipp<'a> {
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

impl<'a> PersistLipp<'a> {
    pub fn new(db: &'a Database) -> Self {
        let root_node = Node::<AddrKey, StateValue>::build_tree_none();
        let bytes = bincode::serialize(&root_node).unwrap();
        let mut tx = db.transaction();
        tx.put(ROCKSDB_COL_ID, &H256::default().as_bytes(), &bytes);
        db.write(tx).unwrap();
        Self {
            roots: vec![H256::default()],
            db,
        }
    }

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

    pub fn apply(&mut self, apply: Apply<AddrKey, StateValue>) {
        self.roots.push(apply.root);
        let mut tx = self.db.transaction();
        for (k, v) in apply.nodes {
            let bytes = bincode::serialize(&v).unwrap();
            tx.put(ROCKSDB_COL_ID, k.as_bytes(), &bytes);
        }
        self.db.write(tx).unwrap();
    }

    pub fn get_latest_root(&self) -> H256 {
        let l = self.roots.len() - 1;
        self.roots[l]
    }

    pub fn get_root_with_version(&self, version: u32) -> H256 {
        // version starts with 1
        match self.roots.get(version as usize) {
            Some(r) => *r,
            None => H256::default(),
        }
    }

    pub fn search(&self, key: AddrKey) -> Option<StateValue> {
        read_tree_without_proof(&self, self.get_latest_root(), key)
    }

    pub fn search_with_proof(&self, key: AddrKey, version: u32) -> (Option<StateValue>, Proof<AddrKey, StateValue>) {
        read_tree(&self, self.get_root_with_version(version), key)
    }

    pub fn insert(&mut self, key: AddrKey, value: StateValue) {
        let immut_ref = unsafe {
            (self as *const PersistLipp).as_ref().unwrap()
        };
        let mut write_context = WriteContext::new(immut_ref, self.get_latest_root());
        write_context.insert(key, value);
        let changes = write_context.changes();
        self.apply(changes);
    }

    pub fn batch_insert(&mut self, inputs: BTreeMap<AddrKey, StateValue>) {
        let immut_ref = unsafe {
            (self as *const PersistLipp).as_ref().unwrap()
        };
        let mut write_context = WriteContext::new(immut_ref, self.get_latest_root());
        for (key, value) in inputs {
            write_context.insert(key, value);
        }
        let changes = write_context.changes();
        self.apply(changes);
    }

    pub fn get_roots_len(&self) -> usize {
        self.roots.len()
    }
}

impl NodeLoader<AddrKey, StateValue> for PersistLipp<'_> {
    fn load_node(&self, id: H256) -> Node<AddrKey, StateValue> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: Node<AddrKey, StateValue> = bincode::deserialize(&node_byte).unwrap();
        node
    }
}

impl NodeLoader<AddrKey, StateValue> for &'_ PersistLipp<'_> {
    fn load_node(&self, id: H256) -> Node<AddrKey, StateValue> {
        let node_byte = self.db.get(ROCKSDB_COL_ID, &id.as_bytes()).unwrap().unwrap();
        let node: Node<AddrKey, StateValue> = bincode::deserialize(&node_byte).unwrap();
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvdb_rocksdb::DatabaseConfig;
    use rand::prelude::*;
    use utils::H160;
    #[test]
    fn test_lipp_persist() {
        let path = "persist_trie";
        if std::path::Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        std::fs::create_dir(&path).unwrap_or_default();
        let mut db_config = DatabaseConfig::with_columns(1);
        db_config.memory_budget.insert(ROCKSDB_COL_ID, 64);
        let db = Database::open(&db_config, path).unwrap();
        let num_of_contract = 10;
        let num_of_address = 10;
        let num_of_versions = 10;
        let mut rng = StdRng::seed_from_u64(1);
        let mut keys = Vec::<AddrKey>::new();
        for _ in 1..=num_of_contract {
            for _ in 1..=num_of_address {
                let addr_key = AddrKey::new(H160::random_using(&mut rng).into(), H256::random_using(&mut rng).into());
                keys.push(addr_key);
            }
        }

        {
            let mut tree = PersistLipp::new(&db);
            let start = std::time::Instant::now();
            for i in 1..=num_of_versions {
                let v = StateValue(H256::from_low_u64_be(i));
                let mut map = BTreeMap::new();
                for key in &keys {
                    map.insert(*key, v);
                }
                tree.batch_insert(map);
            }
            let elapse = start.elapsed().as_nanos();
            println!("avg insert time: {}", elapse / (num_of_address * num_of_contract * num_of_versions) as u128);
            println!("finish insert");
        }

        let tree = PersistLipp::open(&db);
        let mut search_latest = 0;
        let mut search_prove = 0;
        let latest_v = StateValue(H256::from_low_u64_be(num_of_versions));
        for key in &keys {
            let start = std::time::Instant::now();
            let v = tree.search(*key).unwrap();
            let elapse = start.elapsed().as_nanos();
            search_latest += elapse;
            assert_eq!(v, latest_v);
            for i in 1..=num_of_versions {
                let start = std::time::Instant::now();
                let (v, p) = tree.search_with_proof(*key, i as u32);
                let b = verify::<AddrKey, StateValue>(*key, v, tree.get_root_with_version(i as u32), &p);
                let elapse = start.elapsed().as_nanos();
                search_prove += elapse;
                let current_v = StateValue(H256::from_low_u64_be(i));
                assert_eq!(current_v, v.unwrap());
                assert!(b);
            }
        }
        println!("search latest: {}", search_latest / (num_of_address * num_of_contract) as u128);
        println!("search prove: {}", search_prove / (num_of_address * num_of_contract * num_of_versions) as u128);
    }
}
