use serde_json::Value as JsonValue;
use sha3::{Digest, Keccak256};
use super::common::{code::Code, nonce::Nonce};
use primitive_types::{H160, H256, U256};
use std::collections::HashMap;
pub use ethabi::{self, Function, Token};
use std::{fs::File, io::BufReader, path::Path};
use anyhow::{Context as _, Result};
use utils::{types::Address};

// Ref: https://github.com/rust-blockchain/evm/blob/60f4020ab38dc8f21311e44f0f4174192bb1769d/src/executor/stack.rs#L328-L334
pub fn contract_address(creator: Address, nonce: Nonce) -> Address {
    let creator: H160 = H160::from(creator.0.as_fixed_bytes());
    let nonce: U256 = nonce.into();
    let mut stream = rlp::RlpStream::new_list(2);
    stream.append(&creator);
    stream.append(&nonce);
    let address: H160 = H256::from_slice(Keccak256::digest(&stream.out()).as_slice()).into();
    address.into()
}

#[derive(Debug)]
pub struct Contract {
    code: Code,
    funcs: HashMap<String, Function>,
}

impl Contract {
    pub fn from_json_file(file: &Path) -> Result<Self> {
        let reader = BufReader::new(File::open(file)?);
        Self::from_json_value(serde_json::from_reader(reader)?)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_json_value(serde_json::from_slice(data)?)
    }

    pub fn from_json_value(json_data: JsonValue) -> Result<Self> {
        let bytecode = json_data["bytecode"]
            .as_str()
            .context("Failed to read `bytecode`.")?;
        let code = hex::decode(&bytecode[2..])?.into();

        let mut funcs = HashMap::new();
        let abi_data = json_data["abi"]
            .as_array()
            .context("Failed to read `abi`.")?;

        for abi in abi_data {
            if abi["type"] == "function" {
                let func: Function =
                    serde_json::from_value(abi.clone()).context("Failed to decode abi.")?;
                funcs.insert(func.name.clone(), func);
            }
        }

        Ok(Self { code, funcs })
    }

    pub fn code(&self) -> &Code {
        &self.code
    }

    pub fn encode_tx_input(&self, name: &str, args: &[Token]) -> Result<Vec<u8>> {
        self.funcs
            .get(name)
            .context("Failed to find function.")?
            .encode_input(args)
            .context("Failed to encode inputs.")
    }
}