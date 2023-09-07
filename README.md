# COLE: A Column-based Learned Storage for Blockchain Systems
## Components
- `cole-index` and `cole-star` are COLE and COLE with asynchronous merge
- `patricia-trie` is the implementation of the MPT
- `lipp` is the the updatable learned index with node persistence
- `non-learn-cmi` is the column-based Merkle index (CMI) that uses non-learned index
- `exp` is the evaluation backend of all systems including the throughput and the provenance queries

# Install Dependencies
- Install [Rust](https://rustup.rs).
- Run `sudo apt-get update` and `sudo apt install make clang pkg-config libssl-dev`

# Build
- Build the latency testing binary `cargo build --bin latency`
- Build the provenance testing binary `cargo build --bin prov`

# Run Script
`python3 exp/run.py`
