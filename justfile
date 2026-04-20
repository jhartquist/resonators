fixtures:
    uv run scripts/generate_fixtures.py

# Rust
test:
    cargo test

bench:
    cargo bench --bench bank

# Python vs noFFT throughput comparison
bench-vs-nofft:
    uv run scripts/benchmark.py

# Python bindings (requires uv + Rust toolchain)
py-setup:
    cd crates/resonators-py && uv venv --allow-existing && uv pip install maturin pytest numpy

py-build: py-setup
    cd crates/resonators-py && uv run maturin develop --release --uv

py-test: py-build
    cd crates/resonators-py && uv run pytest

# WASM bindings (requires wasm-pack)
wasm-build:
    cd crates/resonators-wasm && wasm-pack build --target web --release --out-name resonators
    sed -i '' 's/"resonators-wasm"/"resonators"/' crates/resonators-wasm/pkg/package.json

# In-browser perf benchmark page (open http://localhost:8000)
wasm-bench: wasm-build
    rm -rf examples/web-bench/pkg
    cp -R crates/resonators-wasm/pkg examples/web-bench/pkg
    cd examples/web-bench && python3 -m http.server 8000

# Live mic spectrogram demo (open http://localhost:8000)
wasm-spectrogram: wasm-build
    rm -rf examples/web-spectrogram/pkg
    cp -R crates/resonators-wasm/pkg examples/web-spectrogram/pkg
    cd examples/web-spectrogram && python3 -m http.server 8000

# Everything
ci: test py-test
