use once_cell::sync::{Lazy, OnceCell};
use rand::{distributions::Uniform, prelude::*};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, prelude::*},
    sync::Mutex,
};
use super::common::{tx_req::TxRequest, nonce::Nonce};
use utils::{types::Address};
use super::eth_utils::{Contract, contract_address, Token};
use anyhow::{anyhow, bail, Result};

pub static YCSB: OnceCell<Mutex<io::BufReader<File>>> = OnceCell::new();
static YCSB_READ_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^READ usertable (\w+) \[.+\]$").unwrap());
static YCSB_UPDATE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^UPDATE usertable (\w+) \[ field\d+=(.+) \]$").unwrap());
static YCSB_WRITE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^INSERT usertable (\w+) \[ field\d+=(.+) \]$").unwrap());
pub static EHT_FILE: OnceCell<Mutex<io::BufReader<File>>> = OnceCell::new();
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContractArg {
    KVStore,
    SmallBank,
    ETH,
}

macro_rules! load_contract {
    ($name: literal) => {{
        static CONTRACT: OnceCell<Contract> = OnceCell::new();
        CONTRACT.get_or_init(|| {
            Contract::from_bytes(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/contracts/build/contracts/",
                $name,
                ".json",
            )))
            .expect("Failed to load the contract.")
        })
    }};
}

impl ContractArg {
    fn get_contract(self) -> &'static Contract {
        match self {
            ContractArg::KVStore => load_contract!("KVstore"),
            ContractArg::SmallBank => load_contract!("SmallBank"),
            ContractArg::ETH => load_contract!("Transfer"),
        }
    }

    fn gen_tx_input(self, rng: &mut impl Rng, n: usize) -> Result<Vec<u8>> {
        match self {
            ContractArg::ETH => {
                let contract = self.get_contract();
                if let Some(eth) = EHT_FILE.get() {
                    let mut eth = eth.lock().map_err(|_e| anyhow!("Failed to lock ETH file."))?;
                    let mut buf = String::new();
                    let buf_len = eth.read_line(&mut buf)?;
                    if buf_len == 0 {
                        bail!("Failed to read ETH file. Reach EOF.");
                    }
                    let buf = buf.trim();
                    let input_vec: Vec<&str> = buf.split(" ").collect();
                    let from_addr = input_vec[0];
                    let to_addr = input_vec[1];
                    let value = input_vec[2];
                    let value_num: u32 = value.parse::<u32>().unwrap();
                    return contract.encode_tx_input("transfer", &[Token::String(from_addr.to_string()), Token::String(to_addr.to_string()), Token::Uint(value_num.into())]);
                } else {
                    bail!("Failed to access ETH file.");
                }
            },
            ContractArg::KVStore => {
                let contract = self.get_contract();

                if let Some(ycsb) = YCSB.get() {
                    let mut ycsb = ycsb.lock().map_err(|_e| anyhow!("Failed to lock YCSB."))?;
                    loop {
                        let mut buf = String::new();
                        let buf_len = ycsb.read_line(&mut buf)?;

                        if buf_len == 0 {
                            bail!("Failed to read ycsb file. Reach EOF.");
                        }

                        let buf = buf.trim();

                        if let Some(cap) = YCSB_READ_RE.captures(&buf) {
                            return contract
                                .encode_tx_input("get", &[Token::String(cap[1].to_string())]);
                        }

                        if let Some(cap) = YCSB_UPDATE_RE.captures(&buf) {
                            return contract.encode_tx_input(
                                "set",
                                &[
                                    Token::String(cap[1].to_string()),
                                    Token::String(cap[2].to_string()),
                                ],
                            );
                        }

                        if let Some(cap) = YCSB_WRITE_RE.captures(&buf) {
                            return contract.encode_tx_input(
                                "set",
                                &[
                                    Token::String(cap[1].to_string()),
                                    Token::String(cap[2].to_string()),
                                ],
                            );
                        }

                        // warn!("Skip line in ycsb file: {}", buf);
                    }
                } else {
                    bail!("Failed to access ycsb file.");
                }
            }
            ContractArg::SmallBank => {
                // https://github.com/ooibc88/blockbench/blob/master/src/macro/smallbank/smallbank.cc
                let op_gen = Uniform::new(1, 7);
                let acc_gen = Uniform::new(1, n as i32);
                let bal_gen = Uniform::new(1, 100);
                let contract = self.get_contract();
                match op_gen.sample(rng) {
                    1 => {
                        // println!("almagate");
                        contract.encode_tx_input(
                        "almagate",
                        &[
                            Token::String(acc_gen.sample(rng).to_string()),
                            Token::String(acc_gen.sample(rng).to_string()),
                        ],
                    )},
                    2 => {
                        // println!("getBalance");
                        contract.encode_tx_input(
                            "getBalance",
                            &[Token::String(acc_gen.sample(rng).to_string())],
                        )
                    },
                    3 => {
                        // println!("updateBalance");
                        contract.encode_tx_input(
                            "updateBalance",
                            &[
                                Token::String(acc_gen.sample(rng).to_string()),
                                Token::Uint(bal_gen.sample(rng).into()),
                            ],
                        )
                    },
                    4 => {
                        // println!("updateSaving");
                        contract.encode_tx_input(
                            "updateSaving",
                            &[
                                Token::String(acc_gen.sample(rng).to_string()),
                                Token::Uint(bal_gen.sample(rng).into()),
                            ],
                        )
                    },
                    5 => {
                        // println!("sendPayment");
                        contract.encode_tx_input(
                            "sendPayment",
                            &[
                                Token::String(acc_gen.sample(rng).to_string()),
                                Token::String(acc_gen.sample(rng).to_string()),
                                Token::Uint(0.into()),
                            ],
                        )
                    },
                    6 => {
                        // println!("writeCheck");
                        contract.encode_tx_input(
                            "writeCheck",
                            &[
                                Token::String(acc_gen.sample(rng).to_string()),
                                Token::Uint(0.into()),
                            ],
                        )
                    } ,
                    _ => unreachable!(),
                }
            }
        }
    }
}

pub fn create_deploy_tx(
    contract: ContractArg,
    caller_address: Address,
    nonce: Nonce,
) -> (Address, TxRequest) {
    let contract_address = contract_address(caller_address, nonce);
    let tx_req = TxRequest::Create {
        nonce: nonce,
        code: contract.get_contract().code().clone(),
    };
    return (contract_address, tx_req);
}

pub fn create_call_tx(
    contract: ContractArg,
    contract_address: Address,
    nonce: Nonce,
    rng: &mut impl Rng,
    n: usize,
) -> TxRequest {
    let tx_req = TxRequest::Call { 
        nonce: nonce, 
        address: contract_address, 
        data: contract.gen_tx_input(rng, n).unwrap(), 
    };
    return tx_req;
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::*;
    use utils::H160;
    #[test]
    fn test_send_tx() {
        let mut rng = StdRng::seed_from_u64(1);
        let caller_address = Address::from(H160::from_low_u64_be(1));
        let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(0));
        println!("contract_address: {:?}, tx_req: {:?}", contract_address, tx_req);
        let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(1));
        println!("contract_address: {:?}, tx_req: {:?}", contract_address, tx_req);
        let (contract_address, tx_req) = create_deploy_tx(ContractArg::SmallBank, caller_address, Nonce::from(2));
        println!("contract_address: {:?}, tx_req: {:?}", contract_address, tx_req);
        for i in 0..10 {
            let call_tx_req = create_call_tx(ContractArg::SmallBank, contract_address, Nonce::from(i+3), &mut rng, 10);
            println!("call_tx_req: {:?}", call_tx_req);
        }
    }

    #[test]
    fn test_read_file() {
        let file_name = "file.dat";
        EHT_FILE.set(Mutex::new(BufReader::new(File::open(file_name).unwrap()))).map_err(|_e| anyhow!("failed to set eth file")).unwrap();
        if let Some(eth) = EHT_FILE.get() {
            loop {
                let mut eth = eth.lock().map_err(|_e| anyhow!("Failed to lock ETH file.")).unwrap();
                let mut buf = String::new();
                let buf_len = eth.read_line(&mut buf).unwrap();
                if buf_len == 0 {
                    println!("reach eof");
                    break;
                }
                let buf = buf.trim();
                let input_vec: Vec<&str> = buf.split(" ").collect();
                let from_addr = input_vec[0];
                let to_addr = input_vec[1];
                let value = input_vec[2];
                let value_num: u32 = value.parse::<u32>().unwrap();
                println!("from: {:?}, to: {:?}, value: {:?}", from_addr, to_addr, value_num);
            }
        } else {
            println!("fail to access ETH file");
        }
    }
}