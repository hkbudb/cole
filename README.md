# COLE: A Column-based Learned Storage for Blockchain Systems
## Components
- `cole-index` and `cole-star` are COLE and COLE with asynchronous merge
- `patricia-trie` is the implementation of the MPT
- `lipp` is the the updatable learned index with node persistence
- `non-learn-cmi` is the column-based Merkle index (CMI) that uses non-learned index
- `exp` is the evaluation backend of all systems including the throughput and the provenance queries

# Install Dependencies
- Install [Rust](https://rustup.rs).
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
- Run `sudo apt update` and `sudo apt install make clang pkg-config libssl-dev libsqlite3-dev`

# Build
- Build the latency testing binary `cargo build --bin latency`
- Build the provenance testing binary `cargo build --bin prov`

# Prepare YCSB Dataset
- Download the latest release of YCSB to the HOME directory:
```
cd ~
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
```
- Use script `build_ycsb.sh` to generate `readonly`, `writeonly`, and `readwriteeven` datasets
```
./exp/build_ycsb.sh
```

Three `txt` files will be generated: `./exp/readonly/readonly-data.txt`, `./exp/writeonly/writeonly-data.txt`, and `./exp/readwriteeven/readwriteeven-data.txt`
# Run Script
`python3 exp/run.py`
