# COLE: A Column-based Learned Storage for Blockchain Systems
## Components
- `cole-index` and `cole-star` are COLE and COLE with asynchronous merge
- `patricia-trie` is the implementation of the MPT
- `lipp` is the the updatable learned index with node persistence
- `non-learn-cmi` is the column-based Merkle index (CMI) that uses non-learned index
- `exp` is the evaluation backend of all systems including the throughput and the provenance queries

## Install Dependencies
- Install [Rust](https://rustup.rs).
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
- Run `sudo apt update` and `sudo apt install make clang pkg-config libssl-dev libsqlite3-dev`

## Build
- Build the latency testing binary `cargo build --release --bin latency`
- Build the provenance testing binary `cargo build --release --bin prov`

## Prepare YCSB Dataset
* Download the latest release of YCSB to the HOME directory:
```
cd ~
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
```
* Install Java
```
sudo apt install default-jdk
sudo apt install default-jre
```
* Use script `build_ycsb.sh` to generate `readonly`, `writeonly`, and `readwriteeven` datasets
```
cd ~/cole-public/exp
./build_ycsb.sh
```

* After the build process finishes, three `txt` files will be generate:
    * `./exp/readonly/readonly-data.txt`
    * `./exp/writeonly/writeonly-data.txt`
    * `./exp/readwriteeven/readwriteeven-data.txt`

* Next, prepare the dataset for provenance queries:
```
cd ~/cole-public/exp/
./build_prov_ycsb.sh
```

* After the build process finishes, a file named `./exp/prov/prov-data.txt` will be generated.

## Run Script

* Use functions like `test_overall_kvstore()`, `test_overall_smallbank()`, and `test_prov()` in `exp/run.py` to evaluate the workload of `KVStore`, `SmallBank`, and provenance query performance.

## Check the Result

The result `json` files can be found in each workload directory (e.g., smallbank, writeonly, prov)

* `*-storage.json` stores the storage information
* `*-ts.json` stores the block timestamp information including start timestamp, end timestamp, and block latency, which can be used to compute the system throughput and latency