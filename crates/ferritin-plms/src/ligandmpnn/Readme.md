# Notes

The initial development sought to create a CLI-equivalent for LigandmMPNN. As of Dec 16 2024, I am no longer pursuing that effort. I'm leavin ghte code here
as an artifact in case it becomes interesting again.


# LigandMPNN-Rust

Rust port of Ligand MPNN.


```sh
bash get_model_params.sh "./model_params"
python scripts/convert_to_safetensor.py
python scripts/convert_to_json.py
#cargo install json_to_rust
#json_to_rust data/pdb_data.json > src/proteintypes.rs
#cat data/pdb_data.json| json_to_rust -j json_object -n MyStruct > out.rs
npm install -g quicktype
quicktype data/pdb_data.json -o output.rs -l rust
```

## Resources

- [Candle](https://github.com/huggingface/candle)
- [Candle Tutorial](https://github.com/ToluClassics/candle-tutorial)
