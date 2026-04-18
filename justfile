fixtures:
    uv run scripts/generate_fixtures.py

# Rust
test:
    cargo test

bench:
    cargo bench --bench bank

# Python bindings (requires uv + Rust toolchain)
py-setup:
    cd crates/resonators-py && uv venv --allow-existing && uv pip install maturin pytest numpy

py-build: py-setup
    cd crates/resonators-py && uv run maturin develop --release --uv

py-test: py-build
    cd crates/resonators-py && uv run pytest

# Everything
ci: test py-test
