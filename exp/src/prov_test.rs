use eth_execution_engine::common::write_trait::BackendWriteTrait;
use eth_execution_engine::send_tx::{create_deploy_tx, create_call_tx, ContractArg, YCSB};
use eth_execution_engine::tx_executor::{exec_tx, test_batch_exec_tx, Backend};
use eth_execution_engine::common::{tx_req::TxRequest, nonce::Nonce};
use eth_execution_engine::{mpt_backend::MPTExecutorBackend, non_learn_cmi_backend::NonLearnCMIBackend, lipp_backend::LIPPBackend, cole_backend::ColeBackend, cole_star_backend::ColeStarBackend};
use utils::{H160, types::{Address, AddrKey}, config::Configs, ROCKSDB_COL_ID};
use rand::prelude::*;
use kvdb_rocksdb::{DatabaseConfig, Database};
use std::collections::{BTreeMap};
use std::{fs::OpenOptions, io::{Read, BufReader, Write}, path::Path, sync::Mutex, str};
use json::{self, object};
use anyhow::{anyhow, Ok, Result};
use patricia_trie::verify_with_addr_key as MPTVerify;
use non_learn_cmi::verify_non_learn_cmi as NonLearnCMIVerify;
use lipp::verify as LIPPVerify;
use cole_index::verify_and_collect_result as ColeVerify;
use cole_star::verify_and_collect_result as ColeStarVerify;
const SAMPLE_SIZE: usize = 200;
const SAMPLE_ITER: usize = 10;

#[derive(Debug, Default, Clone, Copy)]
pub struct ProveVerifyResult {
    pub query_time: usize,
    pub verify_time: usize,
    pub vo_size: usize,
}

impl ProveVerifyResult {
    pub fn add(&mut self, other: &Self) {
        self.query_time += other.query_time;
        self.verify_time += other.verify_time;
        self.vo_size += other.vo_size;
    } 
}

#[derive(Default, Debug)]
pub struct ProvParams {
    pub index_name: String,
    pub scale: usize,
    pub ycsb_path: String,
    pub ycsb_base_row_number: usize,
    pub tx_in_block: usize,
    pub db_path: String,
    pub mem_size: usize,
    pub size_ratio: usize,
    pub epsilon: usize,
    pub mht_fanout: usize,
    pub result_path: String,
    pub fix_window_size: bool,
}

impl ProvParams {
    pub fn from_json_file(path: &str) -> ProvParams {
        let mut file = OpenOptions::new().read(true).open(path).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let json_data = json::parse(data.as_str()).unwrap();
        let mut params = ProvParams::default();
        if json_data.has_key("index_name") {
            params.index_name = json_data["index_name"].to_string();
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
        if json_data.has_key("fix_window_size") {
            params.fix_window_size = json_data["fix_window_size"].as_bool().unwrap();
        }
        return params;
    }
}

pub trait ProvVerifyTrait {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, fanout: usize) -> ProveVerifyResult;
}

impl<'a> ProvVerifyTrait for MPTExecutorBackend<'a> {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, _: usize) -> ProveVerifyResult {
        let mut prove_verify_result = ProveVerifyResult::default();
        let mut results = Vec::new();
        let mut proofs = Vec::new();
        for v in lb..=ub {
            let start = std::time::Instant::now();
            let (r, p) = self.states.search_with_proof(key, v);
            let elapse = start.elapsed().as_nanos();
            prove_verify_result.query_time += elapse as usize;
            let start = std::time::Instant::now();
            let b = MPTVerify(&key, r, self.states.get_root_with_version(v), &p);
            assert!(b);
            let elapse = start.elapsed().as_nanos();
            prove_verify_result.verify_time += elapse as usize;
            results.push(r);
            proofs.push(p);
        }
        let results_size = bincode::serialize(&results).unwrap().len();
        let proof_size = bincode::serialize(&proofs).unwrap().len();
        prove_verify_result.vo_size += results_size + proof_size;
        return prove_verify_result;
    }
}

impl<'a> ProvVerifyTrait for LIPPBackend<'a> {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, _: usize) -> ProveVerifyResult {
        let mut prove_verify_result = ProveVerifyResult::default();
        let mut results = Vec::new();
        let mut proofs = Vec::new();
        for v in lb..=ub {
            let start = std::time::Instant::now();
            let (r, p) = self.states.search_with_proof(key, v);
            let elapse = start.elapsed().as_nanos();
            prove_verify_result.query_time += elapse as usize;
            let start = std::time::Instant::now();
            let b = LIPPVerify(key, r, self.states.get_root_with_version(v), &p);
            assert!(b);
            let elapse = start.elapsed().as_nanos();
            prove_verify_result.verify_time += elapse as usize;
            results.push(r);
            proofs.push(p);
        }
        let results_size = bincode::serialize(&results).unwrap().len();
        let proof_size = bincode::serialize(&proofs).unwrap().len();
        prove_verify_result.vo_size += results_size + proof_size;
        return prove_verify_result;
    }
}

impl<'a> ProvVerifyTrait for NonLearnCMIBackend<'a> {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, _: usize) -> ProveVerifyResult {
        let mut prove_verify_result = ProveVerifyResult::default();
        let start = std::time::Instant::now();
        let (r, p) = self.states.search_with_proof(key, lb, ub);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.query_time += elapse as usize;
        let start = std::time::Instant::now();
        let b = NonLearnCMIVerify(key, lb, ub, &r, self.states.get_root(), &p);
        assert!(b);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.verify_time += elapse as usize;
        let result_size = bincode::serialize(&r).unwrap().len();
        let proof_size = bincode::serialize(&p).unwrap().len();
        prove_verify_result.vo_size += result_size + proof_size;
        return prove_verify_result;
    }
}

impl<'a> ProvVerifyTrait for ColeBackend<'a> {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, fanout: usize) -> ProveVerifyResult {
        let mut prove_verify_result = ProveVerifyResult::default();
        let root = self.states.compute_digest();
        let start = std::time::Instant::now();
        let p = self.states.search_with_proof(key, lb, ub);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.query_time += elapse as usize;
        let start = std::time::Instant::now();
        let (b, _) = ColeVerify(key, lb, ub, root, &p, fanout);
        assert!(b);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.verify_time += elapse as usize;
        let proof_size = bincode::serialize(&p).unwrap().len();
        prove_verify_result.vo_size += proof_size;
        return prove_verify_result;
    }
}

impl<'a> ProvVerifyTrait for ColeStarBackend<'a> {
    fn prove_verify_vo_result(&mut self, key: AddrKey, lb: u32, ub: u32, fanout: usize) -> ProveVerifyResult {
        let mut prove_verify_result = ProveVerifyResult::default();
        let root = self.states.compute_digest();
        let start = std::time::Instant::now();
        let p = self.states.search_with_proof(key, lb, ub);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.query_time += elapse as usize;
        let start = std::time::Instant::now();
        let (b, _) = ColeStarVerify(key, lb, ub, root, &p, fanout);
        assert!(b);
        let elapse = start.elapsed().as_nanos();
        prove_verify_result.verify_time += elapse as usize;
        let proof_size = bincode::serialize(&p).unwrap().len();
        prove_verify_result.vo_size += proof_size;
        return prove_verify_result;
    }
}

pub fn build_db(params: &ProvParams, backend: &mut (impl BackendWriteTrait + Backend)) -> (u32, Vec<AddrKey>) {
    let caller_address = Address::from(H160::from_low_u64_be(1));
    let contract_arg = ContractArg::KVStore;
    let yscb_path = &params.ycsb_path;
    let file = OpenOptions::new().read(true).open(yscb_path).unwrap();
    YCSB.set(Mutex::new(BufReader::new(file))).map_err(|_e| anyhow!("Failed to set YCSB.")).unwrap();
    // deploy contract
    let mut block_id = 1;
    let (contract_address, tx_req) = create_deploy_tx(contract_arg, caller_address, Nonce::from(0));
    exec_tx(tx_req, caller_address, block_id, backend);
    println!("finish deploy contract");
    // run transactions to build db
    let mut rng = StdRng::seed_from_u64(1);
    let tx_in_block = params.tx_in_block;
    let n = params.scale;
    let mut build_requests_per_block = Vec::<TxRequest>::new();
    let mut state = BTreeMap::new();
    let base_row_num = params.ycsb_base_row_number;
    for i in 0..base_row_num {
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
        build_requests_per_block.push(call_tx_req);
        if build_requests_per_block.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            let s = test_batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, backend);
            state.extend(s);
            block_id += 1;
            //clear the requests for the next round
            build_requests_per_block.clear();
        }
    }
    if !build_requests_per_block.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        let s = test_batch_exec_tx(build_requests_per_block.clone(), caller_address, block_id, backend);
        state.extend(s);
        block_id += 1;
        build_requests_per_block.clear();
    }
    println!("finish build base");
    let mut requests = Vec::<TxRequest>::new();
    for i in 0..n {
        let call_tx_req = create_call_tx(contract_arg, contract_address, Nonce::from(i as i32), &mut rng, n as usize);
        requests.push(call_tx_req);
        if requests.len() == tx_in_block {
            // should pack up a block and execute the block
            println!("block id: {}", block_id);
            let s = test_batch_exec_tx(requests.clone(), caller_address, block_id, backend);
            state.extend(s);
            block_id += 1;
            //clear the requests for the next round
            requests.clear();
        }
    }

    if !requests.is_empty() {
        // should pack up a block and execute the block
        println!("block id: {}", block_id);
        let s = test_batch_exec_tx(requests.clone(), caller_address, block_id, backend);
        state.extend(s);
        block_id += 1;
        //clear the requests for the next round
        requests.clear();
    } else {
        block_id -= 1;
    }
    let state_key_vec: Vec<AddrKey> = state.into_keys().collect();
    return (block_id, state_key_vec);
}

pub fn mpt_backend_prov_query(params: &ProvParams) -> Result<()> {
    let result_path = &params.result_path;
    let fixed_window_size = params.fix_window_size;
    let base = format!("{}-{}-{}k-fan{}-fixwindow-{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.fix_window_size);
    let mut prove_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-prove.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();
    let mut backend = MPTExecutorBackend::new(&db);
    let (block_id, state_keys) = build_db(params, &mut backend);
    println!("after build db, block_id: {}", block_id);
    println!("len of roots: {}", backend.states.get_roots_len());
    let mut sample_list = Vec::new();
    for i in 0..SAMPLE_ITER {
        let mut rng = StdRng::seed_from_u64((i+1) as u64);
        let samples: Vec<AddrKey> = state_keys.choose_multiple(&mut rng, SAMPLE_SIZE).cloned().collect();
        sample_list.push(samples);
    }
    let query_window_size = if fixed_window_size {
        vec![32]
    } else {
        vec![2u32, 4, 8, 16, 32, 64, 128]
    };

    for window_size in query_window_size {
        let mut prove_verify_result = ProveVerifyResult::default();
        let lb = block_id - window_size + 1;
        let ub = block_id;
        for samples in &sample_list {
            for key in samples {
                let r = backend.prove_verify_vo_result(*key, lb, ub, 0);
                prove_verify_result.add(&r);
            }
        }
        let query_avg_latency = prove_verify_result.query_time / SAMPLE_SIZE / SAMPLE_ITER;
        let verify_avg_latency = prove_verify_result.verify_time / SAMPLE_SIZE / SAMPLE_ITER;
        let vo_avg_size = prove_verify_result.vo_size / SAMPLE_SIZE / SAMPLE_ITER;

        let record = object! {
            window_size: window_size,
            lb: lb,
            ub: ub,
            query_avg_latency: query_avg_latency,
            verify_avg_latency: verify_avg_latency,
            vo_avg_size: vo_avg_size,
        }.dump();
        write!(prove_file, "{}\n", record).unwrap();
        prove_file.flush().unwrap();
    }
    Ok(())
}

pub fn lipp_backend_prov_query(params: &ProvParams) -> Result<()> {
    let result_path = &params.result_path;
    let fixed_window_size = params.fix_window_size;
    let base = format!("{}-{}-{}k-fan{}-fixwindow-{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.fix_window_size);
    let mut prove_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-prove.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();
    let mut backend = LIPPBackend::new(&db);
    let (block_id, state_keys) = build_db(params, &mut backend);
    println!("after build db, block_id: {}", block_id);
    println!("len of roots: {}", backend.states.get_roots_len());
    let mut sample_list = Vec::new();
    for i in 0..SAMPLE_ITER {
        let mut rng = StdRng::seed_from_u64((i+1) as u64);
        let samples: Vec<AddrKey> = state_keys.choose_multiple(&mut rng, SAMPLE_SIZE).cloned().collect();
        sample_list.push(samples);
    }
    let query_window_size = if fixed_window_size {
        vec![32]
    } else {
        vec![2u32, 4, 8, 16, 32, 64, 128]
    };

    for window_size in query_window_size {
        let mut prove_verify_result = ProveVerifyResult::default();
        let lb = block_id - window_size + 1;
        let ub = block_id;
        for samples in &sample_list {
            for key in samples {
                let r = backend.prove_verify_vo_result(*key, lb, ub, 0);
                prove_verify_result.add(&r);
            }
        }
        let query_avg_latency = prove_verify_result.query_time / SAMPLE_SIZE / SAMPLE_ITER;
        let verify_avg_latency = prove_verify_result.verify_time / SAMPLE_SIZE / SAMPLE_ITER;
        let vo_avg_size = prove_verify_result.vo_size / SAMPLE_SIZE / SAMPLE_ITER;

        let record = object! {
            window_size: window_size,
            lb: lb,
            ub: ub,
            query_avg_latency: query_avg_latency,
            verify_avg_latency: verify_avg_latency,
            vo_avg_size: vo_avg_size,
        }.dump();
        write!(prove_file, "{}\n", record).unwrap();
        prove_file.flush().unwrap();
    }
    Ok(())
}

pub fn non_learn_cmi_backend_prov_query(params: &ProvParams) -> Result<()> {
    let result_path = &params.result_path;
    let fixed_window_size = params.fix_window_size;
    let base = format!("{}-{}-{}k-fan{}-fixwindow-{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.fix_window_size);
    let mut prove_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-prove.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(&db_path).exists() {
        std::fs::remove_dir_all(&db_path).unwrap_or_default();
    }
    let mut db_config = DatabaseConfig::with_columns(1);
    let mem_budget = params.mem_size;
    db_config.memory_budget.insert(ROCKSDB_COL_ID, mem_budget);
    let db = Database::open(&db_config, db_path).unwrap();
    let mut backend = NonLearnCMIBackend::new(&db, params.mht_fanout);
    let (block_id, state_keys) = build_db(params, &mut backend);
    println!("after build db, block_id: {}", block_id);
    let mut sample_list = Vec::new();
    for i in 0..SAMPLE_ITER {
        let mut rng = StdRng::seed_from_u64((i+1) as u64);
        let samples: Vec<AddrKey> = state_keys.choose_multiple(&mut rng, SAMPLE_SIZE).cloned().collect();
        sample_list.push(samples);
    }
    let query_window_size = if fixed_window_size {
        vec![32]
    } else {
        vec![2u32, 4, 8, 16, 32, 64, 128]
    };

    for window_size in query_window_size {
        let mut prove_verify_result = ProveVerifyResult::default();
        let lb = block_id - window_size + 1;
        let ub = block_id;
        for samples in &sample_list {
            for key in samples {
                let r = backend.prove_verify_vo_result(*key, lb, ub, 0);
                prove_verify_result.add(&r);
            }
        }
        let query_avg_latency = prove_verify_result.query_time / SAMPLE_SIZE / SAMPLE_ITER;
        let verify_avg_latency = prove_verify_result.verify_time / SAMPLE_SIZE / SAMPLE_ITER;
        let vo_avg_size = prove_verify_result.vo_size / SAMPLE_SIZE / SAMPLE_ITER;

        let record = object! {
            window_size: window_size,
            lb: lb,
            ub: ub,
            query_avg_latency: query_avg_latency,
            verify_avg_latency: verify_avg_latency,
            vo_avg_size: vo_avg_size,
        }.dump();
        write!(prove_file, "{}\n", record).unwrap();
        prove_file.flush().unwrap();
    }
    Ok(())
}

pub fn cole_backend_prov_query(params: &ProvParams) -> Result<()> {
    let result_path = &params.result_path;
    let fixed_window_size = params.fix_window_size;
    let base = format!("{}-{}-{}k-fan{}-fixwindow-{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.fix_window_size);
    let mut prove_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-prove.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio);
    let mut backend = ColeBackend::new(&configs);
    let (block_id, state_keys) = build_db(params, &mut backend);
    println!("after build db, block_id: {}", block_id);
    let mut sample_list = Vec::new();
    for i in 0..SAMPLE_ITER {
        let mut rng = StdRng::seed_from_u64((i+1) as u64);
        let samples: Vec<AddrKey> = state_keys.choose_multiple(&mut rng, SAMPLE_SIZE).cloned().collect();
        sample_list.push(samples);
    }
    let query_window_size = if fixed_window_size {
        vec![32]
    } else {
        vec![2u32, 4, 8, 16, 32, 64, 128]
    };
    let fanout = params.mht_fanout;
    for window_size in query_window_size {
        let mut prove_verify_result = ProveVerifyResult::default();
        let lb = block_id - window_size + 1;
        let ub = block_id;
        for samples in &sample_list {
            for key in samples {
                let r = backend.prove_verify_vo_result(*key, lb, ub, fanout);
                prove_verify_result.add(&r);
            }
        }
        let query_avg_latency = prove_verify_result.query_time / SAMPLE_SIZE / SAMPLE_ITER;
        let verify_avg_latency = prove_verify_result.verify_time / SAMPLE_SIZE / SAMPLE_ITER;
        let vo_avg_size = prove_verify_result.vo_size / SAMPLE_SIZE / SAMPLE_ITER;

        let record = object! {
            window_size: window_size,
            lb: lb,
            ub: ub,
            query_avg_latency: query_avg_latency,
            verify_avg_latency: verify_avg_latency,
            vo_avg_size: vo_avg_size,
        }.dump();
        write!(prove_file, "{}\n", record).unwrap();
        prove_file.flush().unwrap();
    }
    Ok(())
}

pub fn cole_star_backend_prov_query(params: &ProvParams) -> Result<()> {
    let result_path = &params.result_path;
    let fixed_window_size = params.fix_window_size;
    let base = format!("{}-{}-{}k-fan{}-fixwindow-{}", result_path, params.index_name, params.scale/1000, params.mht_fanout, params.fix_window_size);
    let mut prove_file = OpenOptions::new().create(true).read(true).write(true).truncate(true).open(format!("{}-prove.json", base)).unwrap();
    let db_path = params.db_path.as_str();
    if Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path).unwrap_or_default();
    }
    std::fs::create_dir(db_path).unwrap_or_default();
    // note that here the mem_size is the number of records in the memory, rather than the actual size like 64 MB
    let configs = Configs::new(params.mht_fanout, params.epsilon as i64, db_path.to_string(), params.mem_size, params.size_ratio);
    let mut backend = ColeStarBackend::new(&configs);
    let (block_id, state_keys) = build_db(params, &mut backend);
    println!("after build db, block_id: {}", block_id);
    let mut sample_list = Vec::new();
    for i in 0..SAMPLE_ITER {
        let mut rng = StdRng::seed_from_u64((i+1) as u64);
        let samples: Vec<AddrKey> = state_keys.choose_multiple(&mut rng, SAMPLE_SIZE).cloned().collect();
        sample_list.push(samples);
    }
    let query_window_size = if fixed_window_size {
        vec![32]
    } else {
        vec![2u32, 4, 8, 16, 32, 64, 128]
    };
    let fanout = params.mht_fanout;
    for window_size in query_window_size {
        let mut prove_verify_result = ProveVerifyResult::default();
        let lb = block_id - window_size + 1;
        let ub = block_id;
        for samples in &sample_list {
            for key in samples {
                let r = backend.prove_verify_vo_result(*key, lb, ub, fanout);
                prove_verify_result.add(&r);
            }
        }
        let query_avg_latency = prove_verify_result.query_time / SAMPLE_SIZE / SAMPLE_ITER;
        let verify_avg_latency = prove_verify_result.verify_time / SAMPLE_SIZE / SAMPLE_ITER;
        let vo_avg_size = prove_verify_result.vo_size / SAMPLE_SIZE / SAMPLE_ITER;

        let record = object! {
            window_size: window_size,
            lb: lb,
            ub: ub,
            query_avg_latency: query_avg_latency,
            verify_avg_latency: verify_avg_latency,
            vo_avg_size: vo_avg_size,
        }.dump();
        write!(prove_file, "{}\n", record).unwrap();
        prove_file.flush().unwrap();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_prov_index() {
        let json_file_path = "prov.json";
        let params = ProvParams::from_json_file(json_file_path);
        println!("{:?}", params);
        cole_backend_prov_query(&params).unwrap();
    }
}
