
build:
    cargo build -p ferritin-core


# convert all PSEs to msvj folders
convert: build
   for file in docs/examples/*.pse; do \
        ./target/debug/ferritin-pymol --psefile "$file" --outputdir "${file%.*}"; \
    done


docs: build
    # generate and copy rust docs
    cargo doc --workspace --no-deps
    cp -r target/doc/  docs/doc
    # quarto
    quarto render docs

serve: docs
    quarto preview docs

clean:
    cargo clean -p ferritin-core
    cargo clean --doc
    rm -rf docs/doc/
    rm -rf docs/examples/example


# cargo install cargo-edit
upgrade:
    cargo upgrade


test:
    cargo test

test-full:
    cargo test -- --include-ignored

amplify:
    cargo run --example amplify

amplify-example-01:
    cargo run --example amplify --features metal -- --model-id 350M --protein-string \
    MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL


# ferritin-ligandmpnn was folded into ferritin-plms. The CLI test this recipe
# used to name was commented out from 2024-12 and deleted in ferritin-100.32;
# see src/ligandmpnn/Readme.md if it is ever wanted back.
# ProteinMPNN/LigandMPNN tests, including the ignored (weight-downloading) ones
test-ligandmpnn:
    cargo test -p ferritin-plms --test test_plm_ligandmpnn --test test_ligand_mpnn_loading \
    -- --include-ignored --nocapture

esmc:
    #RUST_BACKTRACE=1 cargo run --example esmc
    cargo run --example esmc


# build every example in the workspace; fails if any does not compile
examples:
    python3 run_examples.py

# also run the examples that terminate on their own (downloads model weights)
examples-run:
    python3 run_examples.py --run

# what would be built and run, without doing it
examples-list:
    python3 run_examples.py --list
