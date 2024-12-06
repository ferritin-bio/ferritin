# cargo build --target wasm32-unknown-unknown --release

cargo +nightly build --target wasm32-unknown-unknown --release

wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
