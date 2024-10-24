cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --out-dir ./out/ --target web target/wasm32-unknown-unknown/release/simpleweb.wasm
python3 -m http.server 8001

# wasm-pack build --target web\
