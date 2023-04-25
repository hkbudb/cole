use eth_execution_engine::common::write_trait::BackendWriteTrait;
use eth_execution_engine::send_tx::{create_deploy_tx, create_call_tx, ContractArg, YCSB};
use eth_execution_engine::tx_executor::{exec_tx, batch_exec_tx, Backend};
use eth_execution_engine::common::{tx_req::TxRequest, nonce::Nonce};
use eth_execution_engine::{mpt_backend::MPTExecutorBackend, non_learn_cmi_backend::NonLearnCMIBackend, lipp_backend::LIPPBackend, cole_backend::ColeBackend, cole_star_backend::ColeStarBackend};
use utils::{H160, types::Address, config::Configs, ROCKSDB_COL_ID};
use rand::prelude::*;
use kvdb_rocksdb::{DatabaseConfig, Database};
use std::{fs::OpenOptions, io::{Read, BufReader, Write}, path::Path, sync::Mutex, process::Command, str};
use json::{self, object};
use anyhow::{anyhow, Ok, Result};
use chrono::prelude::*;

#[derive(Debug, Default)]
pub struct ColeStorageSize {
    pub tree_meta: usize,
    pub level_meta: usize,
    pub state_size: usize,
    pub mht_size: usize,
    pub model_size: usize,
    pub filter_size: usize,
    pub total_size: usize,
}

#[derive(Default, Debug)]
pub struct LatencyParams {
    pub index_name: String,
    pub contract_name: String,
    pub scale: usize,
    pub ycsb_path: String,
    pub ycsb_base_row_number: usize,
    pub num_of_contract: usize,
    pub tx_in_block: usize,
    pub db_path: String,
    pub mem_size: usize,
    pub size_ratio: usize,
    pub epsilon: usize,
    pub mht_fanout: usize,
    pub result_path: String,
}

impl LatencyParams {
    pub fn from_json_file(path: &str) -> LatencyParams {
        let mut file = OpenOptions::new().read(true).open(path).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let json_data = json::parse(data.as_str()).unwrap();
        let mut params = LatencyParams::default();
        if json_data.has_key("index_name") {
            params.index_name = json_data["index_name"].to_string();
        }
        if json_data.has_key("contract_name") {
            params.contract_name = json_data["contract_name"].to_string();
        }
        if json_data.has_key("scale") {
            params.scale = json_data["scale"].as_usize().unwrap();
        }
        if json_data.has_key("ycsb_path") {
            params.ycsb_path = json_data["ycsb_path"].to_string();
        }
        if json_data.has_key("ycsb_base_row_number") {
            params.ycsb_base_row_number = json_data["ycsb_base_row_number"].as_usize().unwrap();
        }
        if json_data.has_key("num_of_contract") {
            params.num_of_contract = json_data["num_of_contract"].as_usize().unwrap();
        }
        if json_data.has_key("tx_in_block") {
            params.tx_in_block = json_data["tx_in_block"].as_usize().unwrap();
        }
        if json_data.has_key("db_path") {
            params.db_path = json_data["db_path"].to_string();
        }
        if json_data.has_key("mem_size") {
            params.mem_size = json_data["mem_size"].as_usize().unwrap();
        }
        if json_data.has_key("size_ratio") {
            params.size_ratio = json_data["size_ratio"].as_usize().unwrap();
        }
        if json_data.has_key("epsilon") {
            params.epsilon = json_data["epsilon"].as_usize().unwrap();
        }
        if json_data.has_key("mht_fanout") {
            params.mht_fanout = json_data["mht_fanout"].as_usize().unwrap();
        }
        if json_data.has_key("result_path") {
            params.result_path = json_data["result_path"].to_string();
        }
        return params;
    }
}

pub fn test_backend_latency(params: &LatencyParams, mut backend: (impl BackendWriteTrait + Backend), index_name: &str) -> Result<()> {
    let caller_address = Address::from(H160::from_low_u64_be(1));
    let contract_name = params.contract_name.as_str();
    let contract_arg;
    if contract_name == "smallbank" {
        contract_arg = ContractArg::SmallBank;
    } else if contract_name == "kvstore" {
        contract_arg = ContractArg::KVStore;
        let yscb_path = &params.ycsb_path;
        let file = OpenOptions::new().read(true).open(yscb_path).unwrap();
        YCSB.set(Mutex::new(BufReader::new(file))).map_err(|_e| anyhow!("Failed to set YCSB.")).unwrap();
    } else {
        return Err(anyhow!("wrong smart contract"));
    }

    // prepare for deploying contracts
    let mut contract_address_list = Vec::<Address>::new();
    let mut block_id = 1;
    let num_of_contract = params.num_of_contract;
    for i in 0..num_of_contract {
        let (contract_address, tx_req) = create_deploy_tx(contract_arg, caller_address, Nonce::from(i));
        contract_address_list.push(contract_address);
        exec_tx(tx_req, caller_address, block_id, &mut backend);
    }
    println!("finish deploy contract");
    // prepare for building the base db for KVSTORE
    let mut rng = StdRng::seed_from_u64(1);
    let tx_in_block = params.tx_in_block;
    let n = params.scale;
    if contract_name == "kvstore" {
        let mut build_requests_per_block = Vec::<TxRequest>::new();
        let base_row_num = params.ycsb_base_row_number;
        for i in 0..base_row_num {
            let contract_id = i % num_of_contract;
            let contract_address = contract_address_list[contract_id as usize];
            let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
            build_requests_per_block.push(call_tx_req);
            if build_requests_per_block.len() == tx_in_block {
                // should pack up a block and execute the block
                println!("block id: {}", block_id);
                batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, &mut backend);
                block_id += 1;
                //clear the requests for the next round
                build_requests_per_block.clear();
            }
        }
        if !build_requests_per_block.is_empty() {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, &mut backend);
            block_id += 1;
            build_requests_per_block.clear();
        }
    }
    backend.flush();
    println!("finish build base");
    // pack up the testing transactions for each block and execute the block
    let mut requests = Vec::<TxRequest>::new();
    let result_path = &params.result_path;
    let base = format!("{}-{}-{}k-fan{}-ratio{}-mem{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.size_ratio, params.mem_size);
    let mut timestamp_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-ts.json", base)).unwrap();
    let mut memory_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-cache.json", base)).unwrap();
    for i in 0..n {
        let contract_id = 0 % num_of_contract;
        let contract_address = contract_address_list[contract_id as usize];
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
        requests.push(call_tx_req);
        if requests.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            let now = Utc::now();
            let start_ts: i64 = now.timestamp_nanos();
            batch_exec_tx(requests.clone(), caller_address, block_id, &mut backend);
            let now = Utc::now();
            let end_ts: i64 = now.timestamp_nanos();
            let elapse = end_ts - start_ts;
            let ts_result_str = object! {
                block_id: block_id,
                start_ts: start_ts,
                end_ts: end_ts,
                elapse: elapse,
            }.dump();
            write!(timestamp_file, "{}\n", ts_result_str).unwrap();
            let index_name = &params.index_name;
            if index_name == "cole" || index_name == "cole_star" {
                let memory_cost = backend.memory_cost();
                let mem_result_str = object! {
                    block_id: block_id,
                    state_cache_size: memory_cost.state_cache_size,
                    model_cache_size: memory_cost.model_cache_size,
                    mht_cache_size: memory_cost.mht_cache_size,
                    filter_size: memory_cost.filter_size,
                }.dump();
                write!(memory_file, "{}\n", mem_result_str).unwrap();
            }
            
            block_id += 1;
            //clear the requests for the next round
            requests.clear();
        }
    }

    if !requests.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        let now = Utc::now();
        let start_ts: i64 = now.timestamp_nanos();
        batch_exec_tx(requests.clone(), caller_address, block_id, &mut backend);
        let now = Utc::now();
        let end_ts: i64 = now.timestamp_nanos();
        let elapse = end_ts - start_ts;
        let ts_result_str = object! {
            block_id: block_id,
            start_ts: start_ts,
            end_ts: end_ts,
            elapse: elapse,
        }.dump();
        write!(timestamp_file, "{}\n", ts_result_str).unwrap();
        requests.clear();
    }
    timestamp_file.flush().unwrap();

    // print the index structure information for cole or cole*
    if index_name == "cole" || index_name == "cole_star" {
        let mut struct_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-struct.txt", base)).unwrap();
        let s = backend.index_stucture_output();
        write!(struct_file, "{}\n", s).unwrap();
        struct_file.flush().unwrap();
    }
    drop(backend);
    return Ok(());
}

pub fn test_mpt_backend_latency(params: &LatencyParams) -> Result<()> {
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();
    
    let backend = MPTExecutorBackend::new(&db);
    test_backend_latency(params, backend, &params.index_name).unwrap();
    Ok(())
}

pub fn test_lipp_backend_latency(params: &LatencyParams) -> Result<()> {
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();
    let backend = LIPPBackend::new(&db);
    test_backend_latency(params, backend, &params.index_name).unwrap();
    Ok(())
}

pub fn test_non_learn_cmi_backend_latency(params: &LatencyParams) -> Result<()> {
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();

    let backend = NonLearnCMIBackend::new(&db, params.mht_fanout);
    test_backend_latency(params, backend, &params.index_name).unwrap();
    Ok(())
}

pub fn test_cole_backend_latency(params: &LatencyParams) -> Result<()> {
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio);
    let backend = ColeBackend::new(&configs);
    test_backend_latency(params, backend, &params.index_name).unwrap();
    Ok(())
}

pub fn test_cole_star_backend_latency(params: &LatencyParams) -> Result<()> {
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio);
    let backend = ColeStarBackend::new(&configs);
    test_backend_latency(params, backend, &params.index_name).unwrap();
    Ok(())
}

pub fn test_index_backend_latency(params: &LatencyParams) -> Result<()> {
    let index_name = &params.index_name;
    if index_name == "mpt" {
        test_mpt_backend_latency(params)
    } else if index_name == "lipp" {
        test_lipp_backend_latency(params)
    } else if index_name == "non_learn_cmi" {
        test_non_learn_cmi_backend_latency(params)
    } else if index_name == "cole" {
        test_cole_backend_latency(params)
    } else if index_name == "cole_star" {
        test_cole_star_backend_latency(params)
    } else {
        Err(anyhow!("wrong index name"))
    }
}

pub fn compute_general_size(path: &str) -> usize {
    let output = Command::new("du")
    .arg("-b")
    .arg("-s")
    .arg(path).output().expect("error");
    let s = String::from_utf8_lossy(&output.stdout);
    let v: Vec<&str> = s.split("\t").collect();
    let size = v[0].parse::<usize>().unwrap();
    return size;
}

pub fn compute_cole_size_breakdown(path: &str) -> ColeStorageSize {
    let mut storage_size = ColeStorageSize::default();
    let path_files = Command::new("ls").arg(path).output().expect("read path files failure");
    let s = str::from_utf8(&path_files.stdout).unwrap();
    let lines: Vec<&str> = s.trim().split("\n").collect();
    let mut tree_meta = 0;
    let mut level_meta = 0;
    let mut state_size = 0;
    let mut mht_size = 0;
    let mut model_size = 0;
    let mut filter_size = 0;

    for line in lines {
        if line == "mht" {
            tree_meta = disk_usage_check_storage(path, line);
        }
        else if line.ends_with("lv") {
            level_meta += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("s_") {
            state_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("m_") {
            model_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("h_") {
            mht_size += disk_usage_check_storage(path, line);
        }
        else if line.starts_with("f_") {
            filter_size += disk_usage_check_storage(path, line);
        }
    }
    storage_size.tree_meta = tree_meta;
    storage_size.level_meta = level_meta;
    storage_size.state_size = state_size;
    storage_size.mht_size = mht_size;
    storage_size.model_size = model_size;
    storage_size.filter_size = filter_size;
    storage_size.total_size = disk_usage_check_storage(path, "");
    return storage_size;
}

fn disk_usage_check_storage(base: &str, file_name: &str) -> usize {
    let du = Command::new("du").arg("-b").arg(format!("{}/{}", base, file_name)).output().unwrap();
    let results: Vec<&str> = str::from_utf8(&du.stdout).unwrap().split("\t").collect();
    results[0].parse::<usize>().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_storage_collection() {
        let storage = compute_cole_size_breakdown("cole_storage");
        println!("{:?}", storage);
        let now = Utc::now();
        let ts: i64 = now.timestamp_nanos();
        println!("current ts is: {}", ts);
        let now = Utc::now();
        let ts: i64 = now.timestamp_nanos();
        println!("current ts is: {}", ts);
    }

    #[test]
    fn test_backends() {
        let json_file_path = "params.json";
        let params = LatencyParams::from_json_file(json_file_path);
        println!("{:?}", params);
        test_index_backend_latency(&params).unwrap();
    }
}
