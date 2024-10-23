use super::common::{code::Code, nonce::Nonce, write_trait::BackendWriteTrait};
use utils::types::{StateKey, StateValue, AddrKey, Address};
use super::common::rw_set::{TxReadData, TxWriteData};
use super::common::tx_req::TxRequest;
use anyhow::{Result, Error, Context as _, ensure};
use evm::backend::Backend as _;
use core::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashSet};
use primitive_types::{H160, H256, U256};
use serde::{Deserialize, Serialize};

pub trait Backend {
    fn get_nonce(&self, acc_address: Address) -> Result<Nonce>;
    fn get_code(&self, acc_address: Address) -> Result<Code>;
    fn get_value(&self, acc_address: Address, key: StateKey) -> Result<StateValue>;
}

struct EVMBackend<'a, B: Backend> {
    backend: &'a B,
    reads: RefCell<TxReadData>,
    error: Cell<Option<Error>>,
}

impl<'a, B: Backend> EVMBackend<'a, B> {
    fn new(backend: &'a B) -> Self {
        Self {
            backend,
            reads: RefCell::new(TxReadData::default()),
            error: Cell::new(None),
        }
    }

    fn set_error(&self, err: Error) {
        self.error.replace(Some(err));
    }

    fn check_error(&self) -> Result<()> {
        match self.error.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn take_reads(&self) -> TxReadData {
        self.reads.replace_with(|_| Default::default())
    }

    fn get_nonce(&self, acc_address: Address) -> Nonce {
        self.reads.borrow_mut().get_or_add_nonce(acc_address, || {
            match self.backend.get_nonce(acc_address) {
                Ok(nonce) => nonce,
                Err(err) => {
                    self.set_error(err);
                    Default::default()
                }
            }
        })
    }

    fn get_code(&self, acc_address: Address) -> Code {
        self.reads
            .borrow_mut()
            .get_or_add_code(acc_address, || match self.backend.get_code(acc_address) {
                Ok(code) => code,
                Err(err) => {
                    self.set_error(err);
                    Default::default()
                }
            })
            .clone()
    }

    fn get_value(&self, acc_address: Address, key: StateKey) -> StateValue {
        self.reads
            .borrow_mut()
            .get_or_add_value(acc_address, key, || {
                match self.backend.get_value(acc_address, key) {
                    Ok(value) => value,
                    Err(err) => {
                        self.set_error(err);
                        Default::default()
                    }
                }
            })
    }
}

impl<'a, B: Backend> evm::backend::Backend for EVMBackend<'a, B> {
    fn gas_price(&self) -> U256 {
        unimplemented!();
    }
    fn origin(&self) -> H160 {
        unimplemented!();
    }
    fn block_hash(&self, _number: U256) -> H256 {
        unimplemented!();
    }
    fn block_number(&self) -> U256 {
        unimplemented!();
    }
    fn block_coinbase(&self) -> H160 {
        unimplemented!();
    }
    fn block_timestamp(&self) -> U256 {
        unimplemented!();
    }
    fn block_difficulty(&self) -> U256 {
        unimplemented!();
    }
    fn block_gas_limit(&self) -> U256 {
        U256::max_value()
    }
    fn block_base_fee_per_gas(&self) -> U256 {
        U256::zero()
    }
    fn chain_id(&self) -> U256 {
        unimplemented!();
    }
    fn exists(&self, _address: H160) -> bool {
        true
    }
    fn basic(&self, address: H160) -> evm::backend::Basic {
        let acc_address: Address = address.into();
        let nonce = self.get_nonce(acc_address);
        evm::backend::Basic {
            balance: U256::max_value(),
            nonce: nonce.into(),
        }
    }
    fn code(&self, address: H160) -> Vec<u8> {
        let acc_address: Address = address.into();
        self.get_code(acc_address).into()
    }
    fn storage(&self, address: H160, index: H256) -> H256 {
        let acc_address: Address = address.into();
        let key: StateKey = index.into();
        self.get_value(acc_address, key).into()
    }
    fn original_storage(&self, address: H160, index: H256) -> Option<H256> {
        Some(self.storage(address, index))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteOutput {
    pub caller: Address,
    pub input: TxRequest,
    pub reads: TxReadData,
    pub writes: TxWriteData,
}

pub fn execute_tx(tx_req: TxRequest, caller: Address, backend: &impl Backend) -> Result<ExecuteOutput> {
    use evm::executor::stack::*;

    let evm_backend = EVMBackend::new(backend);
    let evm_config = evm::Config::istanbul();
    let evm_metadata = StackSubstateMetadata::new(u64::max_value(), &&evm_config);
    let evm_state = MemoryStackState::new(evm_metadata, &evm_backend);
    let mut executor = StackExecutor::new_with_precompiles(evm_state, &evm_config, &());

    let _caller_nonce = Nonce::from(evm_backend.basic(H160::from(caller.0.as_fixed_bytes())).nonce);

    // TODO: disable nonce check for exp.
    // let input_nonce = tx_req.nonce();
    // ensure!(
    //     caller_nonce == input_nonce,
    //     "Invalid nonce (expected: {}, actual: {}).",
    //     caller_nonce,
    //     input_nonce
    // );

    let execute_result = match &tx_req {
        TxRequest::Create { code, .. } => executor.transact_create(
            H160::from(caller.0.as_fixed_bytes()),
            U256::zero(),
            code.clone().into(),
            u64::max_value(),
            Vec::new(),
        ),
        TxRequest::Call { address, data, .. } => {
            executor
                .transact_call(
                    H160::from(caller.0.as_fixed_bytes()),
                    H160::from(address.0.as_fixed_bytes()),
                    U256::zero(),
                    data.clone(),
                    u64::max_value(),
                    Vec::new(),
                )
                .0
        }
    };

    evm_backend
        .check_error()
        .context("Error when accessing the backend.")?;

    ensure!(
        execute_result.is_succeed(),
        "Failed to execute the tx (reason: {:?}).",
        execute_result
    );

    let mut reads = evm_backend.take_reads();
    let mut writes = TxWriteData::default();

    let (applies, _logs) = executor.into_state().deconstruct();
    for apply in applies {
        match apply {
            evm::backend::Apply::Modify {
                address,
                basic,
                code,
                storage,
                reset_storage,
            } => {
                let address = address.into();
                let nonce = Nonce::from(basic.nonce);
                // only record nonce read when the value is updated
                if Some(nonce) == reads.get_nonce(address) {
                    reads.remove_nonce(address);
                } else {
                    writes.add_nonce(address, nonce);
                }
                if let Some(code) = code {
                    writes.add_code(address, Code::from(code));
                }
                if reset_storage {
                    writes.add_reset_values(address);
                }
                for (key, value) in storage {
                    let key = key.into();
                    writes.add_value(address, key, value.into());
                }
            }
            evm::backend::Apply::Delete { address } => {
                let address = address.into();
                writes.delete_account(address);
            }
        }
    }

    Ok(ExecuteOutput {
        caller,
        input: tx_req,
        reads,
        writes,
    })
}

fn update_tx_state(backend: &mut (impl BackendWriteTrait + Backend), writes: &TxWriteData, block_id: u32) {
    for (&acc_addr, acc_data) in writes.iter() {
        for (k, v) in &acc_data.values {
            let addr_key = AddrKey::new(acc_addr.clone(), *k);
            backend.single_write(addr_key, *v, block_id);
        }

        backend.set_acc_nonce(&acc_addr, acc_data.nonce.unwrap_or(backend.get_acc_nonce(&acc_addr)));
        backend.set_acc_code(&acc_addr, acc_data.code.clone().unwrap_or(backend.get_acc_code(&acc_addr)));
    }
}

pub fn exec_tx(tx_req: TxRequest, caller: Address, block_id: u32, backend: &mut (impl BackendWriteTrait + Backend)) {
    let output = execute_tx(tx_req, caller, backend).unwrap();
    update_tx_state(backend, &output.writes, block_id);
}

fn test_batch_update_tx_state(backend: &mut (impl BackendWriteTrait + Backend), multiple_writes: Vec<TxWriteData>, block_id: u32) -> BTreeMap<AddrKey, StateValue> {
    let mut states = BTreeMap::<AddrKey, StateValue>::new();
    for writes in multiple_writes {
        for (&acc_addr, acc_data) in writes.iter() {
            let mut key_set = HashSet::new();
            for (k, v) in &acc_data.values {
                let addr_key = AddrKey::new(acc_addr.clone(), *k);
                if !key_set.contains(&addr_key) {
                    key_set.insert(addr_key.clone());
                    states.insert(addr_key, *v);
                }
            }

            backend.set_acc_nonce(&acc_addr, acc_data.nonce.unwrap_or(backend.get_acc_nonce(&acc_addr)));
            backend.set_acc_code(&acc_addr, acc_data.code.clone().unwrap_or(backend.get_acc_code(&acc_addr)));
        }
    }

    backend.batch_write(states.clone(), block_id);
    return states;
}

pub fn test_batch_exec_tx(requests: Vec<TxRequest>, caller_address: Address, block_id: u32, backend: &mut (impl BackendWriteTrait + Backend))-> BTreeMap<AddrKey, StateValue> {
    let mut multiple_writes: Vec<TxWriteData> = Vec::new();
    for tx_req in requests {
        let output = execute_tx(tx_req, caller_address, backend).unwrap();
        let writes = output.writes;
        multiple_writes.push(writes);
    }
    let states = test_batch_update_tx_state(backend, multiple_writes, block_id);
    return states;
}

fn batch_update_tx_state(backend: &mut (impl BackendWriteTrait + Backend), multiple_writes: Vec<TxWriteData>, block_id: u32) {
    let mut states = BTreeMap::<AddrKey, StateValue>::new();
    for writes in multiple_writes {
        for (&acc_addr, acc_data) in writes.iter() {
            let mut key_set = HashSet::new();
            for (k, v) in &acc_data.values {
                let addr_key = AddrKey::new(acc_addr.clone(), *k);
                if !key_set.contains(&addr_key) {
                    key_set.insert(addr_key.clone());
                    states.insert(addr_key, *v);
                }
            }

            backend.set_acc_nonce(&acc_addr, acc_data.nonce.unwrap_or(backend.get_acc_nonce(&acc_addr)));
            backend.set_acc_code(&acc_addr, acc_data.code.clone().unwrap_or(backend.get_acc_code(&acc_addr)));
        }
    }

    backend.batch_write(states, block_id);
}

pub fn batch_exec_tx(requests: Vec<TxRequest>, caller_address: Address, block_id: u32, backend: &mut (impl BackendWriteTrait + Backend)) {
    let mut multiple_writes: Vec<TxWriteData> = Vec::new();
    for tx_req in requests {
        let output = execute_tx(tx_req, caller_address, backend).unwrap();
        let writes = output.writes;
        multiple_writes.push(writes);
    }
    batch_update_tx_state(backend, multiple_writes, block_id);
}