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
- Run `sudo apt update` and `sudo apt -y install make clang pkg-config libssl-dev libsqlite3-dev llvm m4 build-essential`

## Build
Build the latency testing binary and provenance testing binary
```
cd cole-public
cargo build --release --bin latency
cargo build --release --bin prov
```

## Prepare YCSB Dataset
* Download the latest release of YCSB to the HOME directory:
```
cd ~
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
```
* Install Java
```
sudo apt -y install default-jdk
sudo apt -y install default-jre
```
* Use script `build_ycsb.sh` to generate `readonly`, `writeonly`, and `readwriteeven` datasets
```
cd ~/cole-public/exp
./build_ycsb.sh
```

* After the build process finishes, three `txt` files will be generate:
    * `cole-public/exp/readonly/readonly-data.txt`
    * `cole-public/exp/writeonly/writeonly-data.txt`
    * `cole-public/exp/readwriteeven/readwriteeven-data.txt`

* Next, prepare the dataset for provenance queries:
```
cd ~/cole-public/exp/
./build_prov_ycsb.sh
```

* After the build process finishes, a file named `./exp/prov/prov-data.txt` will be generated.

## Run Script
```
cd ~/cole-public/exp/
python3 run.py
```

* Use functions like `test_overall_kvstore()`, `test_overall_smallbank()`, and `test_prov()` in `cole-public/exp/run.py` to evaluate the workload of `KVStore`, `SmallBank`, and provenance query performance.
* You may select different scales `scale = [1000, 10000, 100000, 1000000, 10000000]` or different indexes `indexes = ["mpt", "cole", "cole_star", "non_learn_cmi"]`

## Check the Result

The result `json` files can be found in each workload directory (e.g., smallbank, writeonly, prov)

* `*-storage.json` stores the storage information
* `*-ts.json` stores the block timestamp information including start timestamp, end timestamp, and block latency, which can be used to compute the system throughput and latency