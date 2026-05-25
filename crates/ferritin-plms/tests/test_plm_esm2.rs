//! Integration smoke tests for the ESM2 Candle ports.

// cargo test test_load_esm2_8m --features metal -- --ignored
// cargo test test_esm2_150m_embedding --features metal -- --ignored

mod support;

use anyhow::Result;
use ferritin_plms::ESM2Models;
use support::model_harness::run_remote_esm2_smoke;

/// Test that we can successfully load the ESM2 650M model
#[test]
#[ignore = "requires downloading model files"]
fn test_load_esm2_8m() -> Result<()> {
    run_remote_esm2_smoke(ESM2Models::T6_8M)
}

/// Test a simple embedding generation using ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_150m_embedding() -> Result<()> {
    run_remote_esm2_smoke(ESM2Models::T30_150M)
}
