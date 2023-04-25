use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use utils::{types::{Address, StateKey, StateValue, AddrKey}};
use super::tx_executor::Backend;
use std::{collections::BTreeMap};
use anyhow::Result;
use kvdb_rocksdb::Database;
use non_learn_cmi::NonLearnCMI;

pub struct NonLearnCMIBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: NonLearnCMI<'a>,
}

impl<'a> NonLearnCMIBackend<'a> {
    pub fn new(db: &'a Database, mb_tree_fanout: usize) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: NonLearnCMI::new(&db, mb_tree_fanout),
        }
    }
}

impl<'a> Backend for NonLearnCMIBackend<'a> {
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
        let addr_key = AddrKey::new(acc_address, key);
        let v = self.states.search_with_latest_version(addr_key);
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

impl<'a> BackendWriteTrait for NonLearnCMIBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, block_id: u32) {
        self.states.insert(addr_key, block_id, v);
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, block_id: u32) {
        let transformed_states: BTreeMap<AddrKey, (u32, StateValue)> = states.into_iter().map(|(k, v)| {
            (k, (block_id, v))
        }).collect();
        self.states.batch_insert(transformed_states);
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
    use rand::prelude::*;
    use utils::{H160, ROCKSDB_COL_ID};
    use kvdb_rocksdb::DatabaseConfig;
    use std::path::Path;
    #[test]
    fn cmi_backend_in_disk() {
        let path = "persist_trie";
        if Path::new(&path).exists() {
            std::fs::remove_dir_all(&path).unwrap_or_default();
        }
        let mut db_config = DatabaseConfig::with_columns(1);
        db_config.memory_budget.insert(ROCKSDB_COL_ID, 64);
        let db = Database::open(&db_config, path).unwrap();
        
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mb_tree_fanout = 10;
        let mut backend = NonLearnCMIBackend::new(&db, mb_tree_fanout);
        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }

        let mut rng = StdRng::seed_from_u64(1);
        let n = 5000;
        let mut requests = Vec::new();
        for i in 0..n {
            let contract_id = i % num_of_contract;
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

        let latest_version = n as u32 / block_size as u32;
        for (k, v) in states {
            let read_v = backend.states.search_with_latest_version(k).unwrap();
            assert_eq!(read_v, v);
            for version in 1..=latest_version {
                let (read_v, p) = backend.states.search_with_proof(k, version, version);
                let b = non_learn_cmi::verify_non_learn_cmi(k, version, version, &read_v, backend.states.get_root(), &p);
                if b == false {
                    println!("false");
                }
            }
            /* let (read_v, p) = backend.states.search_with_proof(&k, block_size as u32 + num_of_contract);
            assert_eq!(read_v.unwrap().0, v.0);
            let b = NonPersistTotalTree::verify_total_tree(&k, block_size as u32 + num_of_contract, read_v, backend.states.get_root(), &p);
            assert!(b); */
        }
    }
}