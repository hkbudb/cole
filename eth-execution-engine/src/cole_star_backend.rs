use super::common::{nonce::Nonce, code::Code, write_trait::BackendWriteTrait};
use utils::{types::{Address, StateKey, StateValue, AddrKey, CompoundKeyValue, CompoundKey}, config::Configs};
use super::tx_executor::Backend;
use std::{cell::UnsafeCell, collections::BTreeMap};
use anyhow::Result;
use cole_star::ColeStar;

pub struct ColeStarBackend<'a> {
    pub nonce_map: BTreeMap<Address, Nonce>,
    pub code_map: BTreeMap<Address, Code>,
    pub states: ColeStar<'a>,
}

impl<'a> ColeStarBackend<'a> {
    pub fn new(configs: &'a Configs) -> Self {
        Self {
            nonce_map: BTreeMap::new(), 
            code_map: BTreeMap::new(), 
            states: ColeStar::new(configs),
        }
    }

    pub fn get_mut_total_tree(&self) -> &'a mut ColeStar<'a> {
        unsafe {
            let const_ptr = &self.states as *const ColeStar;
            let mut_ptr = UnsafeCell::new(const_ptr as *mut ColeStar);
            &mut **mut_ptr.get()
        }
    }
}

impl<'a> Backend for ColeStarBackend<'a> {
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
        let v = self.get_mut_total_tree().search_latest_state_value(addr_key);
        match v {
            Some(v) => {
                Ok(v)
            },
            None => {
                return Ok(StateValue::default());
            }
        }
    }
}

impl<'a> BackendWriteTrait for ColeStarBackend<'a> {
    fn single_write(&mut self, addr_key: AddrKey, v: StateValue, block_id: u32) {
        let compound_key = CompoundKey::new_with_addr_key(addr_key, block_id);
        let compound_key_value = CompoundKeyValue::new_with_compound_key(compound_key, v);
        self.states.insert(compound_key_value);
    }

    fn batch_write(&mut self, states: BTreeMap<AddrKey, StateValue>, block_id: u32) {
        for (addr_key, value) in states {
            let compound_key = CompoundKey::new_with_addr_key(addr_key, block_id);
            let compound_key_value = CompoundKeyValue::new_with_compound_key(compound_key, value);
            self.states.insert(compound_key_value);
        }
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
        self.states.memory_cost()
    }

    fn index_stucture_output(&self,) -> String {
        format!("{:?}", self.states)
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
    use utils::H160;
    use cole_star::verify_and_collect_result;
    #[test]
    fn test_cole_star_backend() {
        let epsilon = 23;
        let fanout = 10;
        let dir_name = "cole_storage";
        if std::path::Path::new(dir_name).exists() {
            std::fs::remove_dir_all(dir_name).unwrap_or_default();
        }
        std::fs::create_dir(dir_name).unwrap_or_default();
        let base_state_num = 3000;
        let size_ratio = 2;
        let configs = Configs::new(fanout, epsilon, dir_name.to_string(), base_state_num, size_ratio);
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let mut backend = ColeStarBackend::new(&configs);

        let num_of_contract = 10;
        let mut contract_address_list = vec![];
        for i in 0..num_of_contract {
            let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(i));
            println!("{:?}", contract_address);
            exec_tx(tx_req, caller_address, i, &mut backend);
            contract_address_list.push(contract_address);
        }
        let mut rng = StdRng::seed_from_u64(1);

        let n = 10000;
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
        let digest = backend.states.compute_digest();
        let latest_version = n as u32 / block_size as u32;
        for (k, v) in states {
            let read_v = backend.states.search_latest_state_value(k).unwrap();
            assert_eq!(read_v, v);
            for version in num_of_contract+1..= num_of_contract+latest_version {
                let p = backend.states.search_with_proof(k, version, version);
                let (b, _) = verify_and_collect_result(k, version, version, digest, &p, fanout);
                if b == false {
                    println!("false");
                }
            }
        }
    }
}