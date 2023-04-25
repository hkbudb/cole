use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use utils::{types::{Address, StateKey, StateValue, AddrKey}};
use super::tx_executor::Backend;
use std::{collections::BTreeMap};
use anyhow::Result;
use patricia_trie::{TestTrie, Key, NibbleBuf, read_trie_without_proof, WriteTrieContext};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TestExecutorBackend {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: TestTrie,
}

impl TestExecutorBackend {
    pub fn new() -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: TestTrie::new(),
        }
    }
}

impl Backend for TestExecutorBackend {
    fn get_code(&self, acc_address: Address) -> Result<Code> {
        if self.code_map.contains_key(&acc_address) {
            Ok(self.code_map.get(&acc_address).unwrap().clone())
        } else {
            Ok(Code::default())
        }
    }

    fn get_nonce(&self, acc_address: Address) -> Result<Nonce> {
        if self.nonce_map.contains_key(&acc_address) {
            return Ok(self.nonce_map.get(&acc_address).unwrap().clone());
        } else {
            return Ok(Nonce::default());
        }
    }

    fn get_value(&self, acc_address: Address, key: StateKey) -> Result<StateValue> {
        let k = Key(NibbleBuf::from_h160_and_h256(acc_address.0, key.0));
        let v = read_trie_without_proof(&self.states, self.states.get_latest_root(), &k).unwrap();
        match v {
            Some(v) => {
                Ok(StateValue(v.0))
            },
            None => {
                return Ok(StateValue::default());
            }
        }
    }
}

impl BackendWriteTrait for TestExecutorBackend {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, _: u32) {
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(&self.states, self.states.get_latest_root());
        let k = Key(NibbleBuf::from_addr_key(addr_key));
        ctx.insert(&k, v).unwrap();
        let changes = ctx.changes();
        self.states.apply(changes);
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, _: u32) {
        let mut ctx: WriteTrieContext<Key, _, _> = WriteTrieContext::new(&self.states, self.states.get_latest_root());
        for (addr_key, v) in states.into_iter() {
            let k = Key(NibbleBuf::from_addr_key(addr_key));
            ctx.insert(&k, v).unwrap();
        }
        let changes = ctx.changes();
        self.states.apply(changes);
    }

    fn set_acc_nonce(&mut self, contract_addr: &Address, contract_nonce: Nonce) {
        self.nonce_map.insert(*contract_addr, contract_nonce);
    }

    fn get_acc_nonce(&self, contract_addr: &Address) -> Nonce {
        match self.nonce_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Nonce::default()
            }
        }
    }

    fn set_acc_code(&mut self, contract_addr: &Address, contract_code: Code) {
        self.code_map.insert(*contract_addr, contract_code);
    }

    fn get_acc_code(&self, contract_addr: &Address) -> Code {
        match self.code_map.get(contract_addr) {
            Some(r) => {
                r.clone()
            },
            None => {
                Code::default()
            }
        }
    }

    fn memory_cost(&self,) -> cole_index::MemCost {
        todo!()
    }

    fn index_stucture_output(&self,) -> String {
        todo!()
    }

    fn flush(&self) {
        
    }
}

#[cfg(test)]
mod tests {
    use crate::send_tx::{create_deploy_tx, create_call_tx, ContractArg};
    use super::super::tx_executor::{exec_tx, test_batch_exec_tx};
    use super::super::common::tx_req::TxRequest;
    use super::*;
    // use patricia_trie::{read_trie, verify_trie_proof};
    use rand::prelude::*;
    use utils::H160;

    #[test]
    fn test_backend_in_memory() {
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = TestExecutorBackend::new();
        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            // backend.contract_code_map.insert(contract_address, Code::default());
            // backend.contract_nonce_map.insert(contract_address, Nonce::from(0));
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let n = 100000;

        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = 0 % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
            requests.push(call_tx_req);
        }

        let block_size = 100;
        let blocks: Vec<Vec<TxRequest>> = requests.chunks(block_size).into_iter().map(|v| v.to_owned()).collect();
        let mut i = 1;
        let mut states = BTreeMap::<AddrKey, StateValue>::new();
        let start = std::time::Instant::now();
        for block in blocks {
            let s = test_batch_exec_tx(block, caller_address, i + num_of_contract, &mut backend);
            states.extend(s);
            i += 1;
        }
        let elapse = start.elapsed().as_nanos();
        println!("time: {}", elapse / n as u128);

        // let latest_version = n as u32 / block_size as u32;
        for (k, v) in &states {
            let search_k = Key(NibbleBuf::from_addr_key(*k));
            let read_v = read_trie_without_proof(&backend.states, backend.states.get_latest_root(), &search_k).unwrap().unwrap();
            assert_eq!(read_v, *v);
            /* for version in 1..=latest_version {
                let (read_v, p) = read_trie(&backend.states, backend.states.get_root_with_version(version), &search_k).unwrap();
                assert_eq!(read_v.unwrap().0, v.0);
                let b = verify_trie_proof(&search_k, read_v, backend.states.get_root_with_version(version), &p);
                assert!(b);
            } */
        }
    }
}