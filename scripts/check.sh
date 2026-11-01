#!/bin/bash
# The checks CI runs, all feature sets. From the repository root; prints ALL-OK.
set -eu
cargo fmt --all --check
cargo clippy -q --all-targets --all-features -- -D warnings
cargo clippy -q --all-targets --no-default-features -- -D warnings
cargo test -q --all-features
cargo test -q --no-default-features
RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --all-features
RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --no-default-features
cargo +1.95 check -q --all-features --all-targets
echo ALL-OK
