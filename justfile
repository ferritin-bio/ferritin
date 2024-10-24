
build:
    cargo build


# convert all PSEs to msvj folders
convert: build
    for file in docs/examples/*.pse; do \
        ./target/debug/pseutils --psefile "$file" --outputdir "${file%.*}"; \
    done


docs: convert
    # generate and copy rust docs
    cargo doc --no-deps
    cp -r target/doc  docs/
    # quarto
    quarto render docs

serve: docs
    quarto preview docs

clean:
    cargo clean --doc
    rm -rf docs/doc/
    rm -rf docs/examples/example
