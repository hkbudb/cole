extern crate locallib;
use locallib::prov_test::{ProvParams, mpt_backend_prov_query, lipp_backend_prov_query, non_learn_cmi_backend_prov_query, cole_backend_prov_query, cole_star_backend_prov_query};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    let json_file_path = args.last().unwrap();
    let params = ProvParams::from_json_file(json_file_path);
    println!("{:?}", params);
    let index_name = &params.index_name;
    if index_name == "mpt" {
        mpt_backend_prov_query(&params).unwrap();
    } else if index_name == "lipp" {
        lipp_backend_prov_query(&params).unwrap();
    } else if index_name == "non_learn_cmi" {
        non_learn_cmi_backend_prov_query(&params).unwrap();
    } else if index_name == "cole" {
        cole_backend_prov_query(&params).unwrap();
    } else if index_name == "cole_star" {
        cole_star_backend_prov_query(&params).unwrap();
    }
}