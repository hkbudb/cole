use std::cmp::{Ordering, PartialOrd, Ord};
use std::collections::BinaryHeap;
use crate::pager::mht_pager::{StreamMHTConstructor, HashPageWriter};
use crate::pager::model_pager::{StreamModelConstructor, ModelPageWriter};
use crate::pager::state_pager::{StateIterator, StatePageWriter, InMemStateIterator};
use crate::types::{CompoundKeyValue, Digestible};
use crate::{H160, H256};
use growable_bloom_filter::GrowableBloom;

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct MergeElement {
    pub state: CompoundKeyValue, // state
    pub i: usize, // the index of the file
}

impl PartialOrd for MergeElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return other.state.partial_cmp(&self.state);
    }
}

impl Ord for MergeElement {
    fn cmp(&self, other: &MergeElement) -> Ordering {
        return other.state.cmp(&self.state);
    }
}

pub fn merge(mut inputs: Vec<StateIterator>, output_state_file_name: &str, output_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriter, ModelPageWriter, HashPageWriter, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriter::create(output_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
    let mut mht_constructor = StreamMHTConstructor::new(output_mht_file_name, fanout);
    let mut minheap = BinaryHeap::<MergeElement>::new();
    let k = inputs.len();
    
    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = CompoundKeyValue::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into(), 0, H256::from_low_u64_be(0).into());
    let max_state = CompoundKeyValue::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX, H256::from_low_u64_be(0).into());
    // add the state's key to the model constructor
    // model_constructor.append_state_key(&min_state.key);
    // add the state's hash to the mht constructor
    mht_constructor.append_hash(min_state.to_digest());
    // add the smallest state to the writer
    state_writer.append(min_state);

    // add the first states from each iterator
    for i in 0..k {
        let r = inputs[i].next().unwrap();
        let elem = MergeElement {
            state: r,
            i,
        };
        minheap.push(elem);
    }
    // flag of number of full iterators
    let mut full_cnt = 0;
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            model_constructor.append_state_key(&state.key);
            // insert the state's key to the bloom filter
            if filter.is_some() {
                let addr_key = state.key.addr;
                filter.as_mut().unwrap().insert(addr_key);
            }
            // add the state's hash to the mht constructor
            mht_constructor.append_hash(state.to_digest());
            // add the smallest state to the writer
            state_writer.append(state);
        }

        let i = elem.i;
        // initiate a state with the maximum value
        let mut state = CompoundKeyValue::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX, H256::from_low_u64_be(0).into());
        let r = inputs[i].next();
        if r.is_some() {
            // load the next smallest state from the iterator
            state = r.unwrap();
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
        // create a new merge element with the previously loaded next smallest state
        let elem = MergeElement {
            state,
            i,
        };
        // push the element to the heap
        minheap.push(elem);
    }

    // add the max state as the upper boundary to help the completeness check
    // add the state's hash to the mht constructor
    mht_constructor.append_hash(max_state.to_digest());
    // add the max state to the writer
    state_writer.append(max_state);
    
    // flush the state writer
    state_writer.flush();
    // finalize the model constructor
    model_constructor.finalize_append();
    mht_constructor.build_mht();
    return (state_writer, model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
}

pub fn in_memory_merge(mut inputs: Vec<InMemStateIterator>, output_state_file_name: &str, output_model_file_name: &str, output_mht_file_name: &str, epsilon: i64, fanout: usize, mut filter: Option<GrowableBloom>) -> (StatePageWriter, ModelPageWriter, HashPageWriter, Option<GrowableBloom>) {
    let mut state_writer = StatePageWriter::create(output_state_file_name);
    let mut model_constructor = StreamModelConstructor::new(output_model_file_name, epsilon);
    let mut mht_constructor = StreamMHTConstructor::new(output_mht_file_name, fanout);
    let mut minheap = BinaryHeap::<MergeElement>::new();
    let k = inputs.len();

    // before adding the actual state, add a min_boundary state to help the completeness check
    let min_state = CompoundKeyValue::new(H160::from_low_u64_be(0).into(), H256::from_low_u64_be(0).into(), 0, H256::from_low_u64_be(0).into());
    let max_state = CompoundKeyValue::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX, H256::from_low_u64_be(0).into());
    // add the state's key to the model constructor
    // model_constructor.append_state_key(&min_state.key);
    // add the state's hash to the mht constructor
    mht_constructor.append_hash(min_state.to_digest());
    // add the smallest state to the writer
    state_writer.append(min_state);

    // add the first states from each iterator
    for i in 0..k {
        let r = inputs[i].next().unwrap();
        let elem = MergeElement {
            state: CompoundKeyValue { key: r.0, value: r.1 },
            i,
        };
        minheap.push(elem);
    }
    // flag of number of full iterators
    let mut full_cnt = 0;
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // avoid duplication of adding the min and max state
        if state != min_state && state != max_state {
            // add the state's key to the model constructor
            model_constructor.append_state_key(&state.key);
            // insert the state's key to the bloom filter
            if filter.is_some() {
                let addr_key = state.key.addr;
                filter.as_mut().unwrap().insert(addr_key);
            }
            // add the state's hash to the mht constructor
            mht_constructor.append_hash(state.to_digest());
            // add the smallest state to the writer
            state_writer.append(state);
        }

        let i = elem.i;
        // initiate a state with the maximum value
        let mut state = CompoundKeyValue::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX, H256::from_low_u64_be(0).into());
        let r = inputs[i].next();
        if r.is_some() {
            // load the next smallest state from the iterator
            let k_v = r.unwrap();
            state = CompoundKeyValue { key: k_v.0, value: k_v.1 };
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
        // create a new merge element with the previously loaded next smallest state
        let elem = MergeElement {
            state,
            i,
        };
        // push the element to the heap
        minheap.push(elem);
    }

    // add the max state as the upper boundary to help the completeness check
    // add the state's hash to the mht constructor
    mht_constructor.append_hash(max_state.to_digest());
    // add the max state to the writer
    state_writer.append(max_state);

    // flush the state writer
    state_writer.flush();
    // finalize the model constructor
    model_constructor.finalize_append();
    mht_constructor.build_mht();
    return (state_writer, model_constructor.output_model_writer, mht_constructor.output_mht_writer, filter);
}

pub fn merge_sort_states(mut inputs: Vec<StateIterator>, output_file_name: &str) -> StatePageWriter {
    let mut pager = StatePageWriter::create(output_file_name);
    let mut minheap = BinaryHeap::<MergeElement>::new();
    let k = inputs.len();
    // add the first states from each iterator
    for i in 0..k {
        let r = inputs[i].next().unwrap();
        let elem = MergeElement {
            state: r,
            i,
        };
        minheap.push(elem);
    }
    // flag of number of full iterators
    let mut full_cnt = 0;
    while full_cnt < k {
        // pop the smallest state from the heap
        let elem = minheap.pop().unwrap();
        let state = elem.state;
        // add the smallest state to the pager streaminly
        pager.append(state);
        let i = elem.i;
        // initiate a state with the maximum value
        let mut state = CompoundKeyValue::new(H160::from_slice(&vec![255u8; 20]).into(), H256::from_slice(&vec![255u8; 32]).into(), u32::MAX, H256::from_low_u64_be(0).into());
        let r = inputs[i].next();
        if r.is_some() {
            // load the next smallest state from the iterator
            state = r.unwrap();
        } else {
            // the iterator reaches the last
            full_cnt += 1;
        }
        // create a new merge element with the previously loaded next smallest state
        let elem = MergeElement {
            state,
            i,
        };
        // push the element to the heap
        minheap.push(elem);
    }
    // flush the pager
    pager.flush();
    return pager;
}

#[cfg(test)]
mod tests {
    use std::{slice::Iter};
    use crate::{OpenOptions, cacher::CacheManager};
    use rand::{rngs::StdRng, SeedableRng};
    use crate::{pager::{MAX_NUM_STATE_IN_PAGE, PAGE_SIZE}, models::CompoundKeyModel};

    use super::*;
    #[derive(PartialEq, Eq, Clone, Copy)]
    struct T(u64);
    impl PartialOrd for T {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            return other.0.partial_cmp(&self.0);
        }
    }
    
    impl Ord for T {
        fn cmp(&self, other: &Self) -> Ordering {
            return other.0.cmp(&self.0);
        }
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    struct Element {
        pub state: u64, // state
        pub i: usize, // the index of the file
    }

    impl PartialOrd for Element {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            return other.state.partial_cmp(&self.state);
        }
    }
    
    impl Ord for Element {
        fn cmp(&self, other: &Self) -> Ordering {
            return other.state.cmp(&self.state);
        }
    }

    fn merge_sort(mut inputs: Vec<Iter<u64>>) -> Vec<u64> {
        let mut v = Vec::<u64>::new();
        let mut minheap = BinaryHeap::<Element>::new();
        let k = inputs.len();
        for i in 0..k {
            let r = inputs[i].next().unwrap();
            let elem = Element {
                state: *r,
                i,
            };
            minheap.push(elem);
        }

        let mut full_cnt = 0;
        while full_cnt < k {
            println!("heap size: {}", minheap.len());
            let elem = minheap.pop().unwrap();
            println!("elem: {:?}", elem);
            v.push(elem.state);
            let i = elem.i;
            let mut state = u64::MAX;
            let r = inputs[i].next();
            if r.is_some() {
                state = *r.unwrap();
            } else {
                full_cnt += 1;
            }

            let elem = Element {
                state,
                i,
            };
            minheap.push(elem);
        }
        return v;
    }

    #[test]
    fn test_pager_merge_sort() {
        let k: usize = 10;
        let n: usize = 100000;
        let mut pagers = Vec::<StatePageWriter>::new();
        let mut total_states = Vec::<CompoundKeyValue>::new();
        for i in 0..k {
            let mut pager = StatePageWriter::create(&format!("data{}.dat", i));
            let mut state_vec = Vec::<CompoundKeyValue>::new();
            for i in 0..n {
                let acc_addr = H160::random();
                let state_addr = H256::random();
                let version = i as u32;
                let value = H256::random();
                let state = CompoundKeyValue::new(acc_addr.into(), state_addr.into(), version, value.into());
                state_vec.push(state.clone());
                total_states.push(state);
            }
            state_vec.sort();
            for s in state_vec {
                pager.append(s);
            }
            pager.flush();
            pagers.push(pager);
        }
        total_states.sort();
        let mut iters = Vec::<StateIterator>::new();
        while !pagers.is_empty() {
            let it = pagers.remove(0).to_state_iter();
            iters.push(it);
        }
        let out_page_writer = merge_sort_states(iters, "out-data.dat");
        let mut out_page_reader = out_page_writer.to_state_reader();
        let mut cache_manager = CacheManager::new();
        for i in 0..n*k {
            let page_id = i / MAX_NUM_STATE_IN_PAGE;
            let inner_page_pos = i % MAX_NUM_STATE_IN_PAGE;
            let v = out_page_reader.read_deser_page_at(0, page_id, &mut cache_manager);
            let state = v[inner_page_pos];
            assert_eq!(state, total_states[i]);
        }
    }
    #[test]
    fn test_simple_merge_sort() {
        let vecs: Vec<Vec<u64>> = vec![vec![2, 6, 12, 34], vec![1, 9, 20, 1000, 3010, 4000], vec![23, 34, 90, 2000, 3000]];
        let inputs: Vec<_> = vecs.iter().map(|v| v.iter()).collect();
        let r = merge_sort(inputs);
        println!("{:?}", r);
    }

    #[test]
    fn test_min_heap() {
        let mut minheap = BinaryHeap::<T>::new();
        minheap.push(T(2));
        minheap.push(T(1));
        minheap.push(T(42));
        while let Some(T(root)) = minheap.pop() {
            println!("{:?}", root);
        }
    }

    #[test]
    fn generate_states() {
        let k: usize = 10;
        let n: usize = 1000;
        let mut rng = StdRng::seed_from_u64(1);
        let mut pagers = Vec::<StatePageWriter>::new();
        for i in 0..k {
            let mut pager = StatePageWriter::create(&format!("data{}.dat", i));
            let mut state_vec = Vec::<CompoundKeyValue>::new();
            for i in 0..n {
                let acc_addr = H160::random_using(&mut rng);
                let state_addr = H256::random_using(&mut rng);
                let version = i as u32;
                let value = H256::random_using(&mut rng);
                let state = CompoundKeyValue::new(acc_addr.into(), state_addr.into(), version, value.into());
                state_vec.push(state.clone());
            }
            state_vec.sort();
            for s in state_vec {
                pager.append(s);
            }
            pager.flush();
            pagers.push(pager);
        }
    }

    #[test]
    fn test_total_merge() {
        let k = 10;
        let n = 1000;
        let epsilon = 2;
        let fanout = 2;
        let mut iters = Vec::<StateIterator>::new();
        for i in 0..k {
            let file = OpenOptions::new().create(true).read(true).write(true).open(&format!("data{}.dat", i)).unwrap();
            let iter = StateIterator::create_with_num_states(file, n);
            iters.push(iter);
        }
        let filter = Some(GrowableBloom::new(0.1, n as usize));
        let r = merge(iters, "out_state.dat", "out_model.dat", "out_mht.dat", epsilon, fanout, filter);
        let (out_state_writer, out_model_writer, out_mht_writer, _) = r;
        let (mut out_state_reader, mut out_model_reader, mut out_mht_reader) = (out_state_writer.to_state_reader(), out_model_writer.to_model_reader(), out_mht_writer.to_hash_reader());
        let state_page_num = out_state_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        let mut states = Vec::<CompoundKeyValue>::new();
        let mut cache_manager = CacheManager::new();
        for page_id in 0..state_page_num {
            let v = out_state_reader.read_deser_page_at(0, page_id, &mut cache_manager);
            states.extend(&v);
        }
        println!("states len: {}", states.len());
        let mut models = Vec::<CompoundKeyModel>::new();
        let model_page_num = out_model_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..model_page_num {
            let v = out_model_reader.read_deser_page_at(0, page_id, &mut cache_manager).v;
            models.extend(&v);
        }
        println!("model len: {}", models.len());

        let mut hashes = Vec::<H256>::new();
        let hash_page_num = out_mht_reader.file.metadata().unwrap().len() as usize / PAGE_SIZE;
        for page_id in 0..hash_page_num {
            let v = out_mht_reader.read_deser_page_at(0, page_id, &mut cache_manager);
            hashes.extend(&v);
        }
        println!("hash len: {}", hashes.len());
        // println!("sleep");
    }
}