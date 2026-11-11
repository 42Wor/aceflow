//! Shared native foundation for Mallow projects.

#[cfg(feature = "extension-module")]
mod bindings;
pub mod cpu;
pub mod device;
pub mod dsp;
pub mod error;
pub mod nn;
pub mod tensor;
pub mod tokenizer;
#[cfg(feature = "wgpu-runtime")]
pub mod wgpu;

pub use cpu::AceEngine;
pub use cpu::beam_search::{BeamSearch, BeamSearchConfig, BeamSearchResult};
pub use cpu::rnn::{RnnKind, RnnOutput, rnn_forward as rnn_forward_packed};
pub use cpu::seq_attn::{AttentionKind, seq2seq_attention};
pub use device::{default_device, set_default_device, Device};
pub use error::{Error, Result};
pub use tokenizer::{AutoTokenizer, FastTokenizer, PyAutoTokenizer};
#[cfg(feature = "wgpu-runtime")]
pub use wgpu::WgpuAceEngine;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AceEngine>()?;
    #[cfg(feature = "wgpu-runtime")]
    module.add_class::<WgpuAceEngine>()?;
    module.add_class::<FastTokenizer>()?;
    module.add_class::<PyAutoTokenizer>()?;
    bindings::register(module)?;

    module.add("SAMPLE_RATE", dsp::SAMPLE_RATE)?;
    module.add("N_FFT", dsp::N_FFT)?;
    module.add("HOP_LENGTH", dsp::HOP_LENGTH)?;
    module.add("N_MELS", dsp::N_MELS)?;
    module.add("N_SAMPLES", dsp::N_SAMPLES)?;
    module.add("N_FRAMES", dsp::N_FRAMES)?;
    module.add("N_FREQS", dsp::N_FREQS)?;
    module.add("PAD_ID", tokenizer::PAD_ID)?;
    module.add("BOS_ID", tokenizer::BOS_ID)?;
    module.add("EOS_ID", tokenizer::EOS_ID)?;
    module.add("BYTE_OFFSET", tokenizer::BYTE_OFFSET)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
