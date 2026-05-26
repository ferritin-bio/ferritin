// Model names
pub const ESM3_OPEN_SMALL: &str = "esm3_sm_open_v1";
pub const ESM3_OPEN_SMALL_ALIAS_1: &str = "esm3-open-2024-03";
pub const ESM3_OPEN_SMALL_ALIAS_2: &str = "esm3-sm-open-v1";
pub const ESM3_OPEN_SMALL_ALIAS_3: &str = "esm3-open";
pub const ESM3_STRUCTURE_ENCODER_V0: &str = "esm3_structure_encoder_v0";
pub const ESM3_STRUCTURE_DECODER_V0: &str = "esm3_structure_decoder_v0";
pub const ESM3_FUNCTION_DECODER_V0: &str = "esm3_function_decoder_v0";
pub const ESMC_600M: &str = "esmc_600m";
pub const ESMC_300M: &str = "esmc_300m";

pub fn model_is_locally_supported(x: &str) -> bool {
    matches!(
        x,
        ESM3_OPEN_SMALL
            | ESM3_OPEN_SMALL_ALIAS_1
            | ESM3_OPEN_SMALL_ALIAS_2
            | ESM3_OPEN_SMALL_ALIAS_3
    )
}

pub fn normalize_model_name(x: &str) -> &str {
    if matches!(
        x,
        ESM3_OPEN_SMALL_ALIAS_1 | ESM3_OPEN_SMALL_ALIAS_2 | ESM3_OPEN_SMALL_ALIAS_3
    ) {
        ESM3_OPEN_SMALL
    } else {
        x
    }
}
