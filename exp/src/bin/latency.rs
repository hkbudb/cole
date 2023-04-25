extern crate locallib;
use locallib::latency_test::{LatencyParams, test_index_backend_latency};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    let json_file_path = args.last().unwrap();
    let params = LatencyParams::from_json_file(json_file_path);
    println!("{:?}", params);
    test_index_backend_latency(&params).unwrap();
}