# Build and test the core Rust crate
test:
    cargo test -p resonators

# Build the core crate in release mode
build:
    cargo build -p resonators --release

# Build Python bindings and install into notebooks venv
build-py:
    cd crates/resonators-py && VIRTUAL_ENV={{justfile_directory()}}/notebooks/.venv maturin develop --release --uv

# Build WASM package with SIMD
build-wasm:
    cd crates/resonators-wasm && RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir pkg --release

# Generate reference test data using noFFT C++ (macOS only)
generate-reference:
    cd notebooks && uv run python generate_reference.py

# Run comparison notebook
run-notebook:
    cd notebooks && uv run jupyter notebook comparison.ipynb

# Run criterion benchmarks
bench:
    cargo bench -p resonators

# Serve WASM example locally
serve-wasm:
    cd crates/resonators-wasm && python3 -m http.server 8787

# Clean all build artifacts
clean:
    cargo clean
    rm -rf crates/resonators-wasm/pkg
