#!/bin/bash
set -e  # Exit on error

# cargo clean
mkdir -p target/wasm32-unknown-unknown/release
cargo update
cargo +nightly build \
    --target wasm32-unknown-unknown \
    --release \
    -Z build-std=std,panic_abort \
    --no-default-features



# cargo build --target wasm32-unknown-unknown --release
# rustup +nightly target add wasm32-unknown-unknown
# cargo +nightly build --target wasm32-unknown-unknown --release
#
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
