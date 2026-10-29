//! Baseline (a): one monolithic n-gram model over byte-level BPE tokens.
//!
//! The BPE tokenizer is trained on the same training files as Sector-LM. The n-gram
//! model is the same interpolated Kneser-Ney implementation used inside each sector
//! (`SectorKneserNeyModel`), with a single model for all tokens, so the comparison
//! differs only in tokenization and partitioning, not in smoothing.

use rayon::prelude::*;
use std::collections::HashMap;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{Tokenizer, TokenizerBuilder};

use crate::model::ngram::SectorKneserNeyModel;

pub struct BpeNgramBaseline {
    pub tokenizer: Tokenizer,
    pub model: SectorKneserNeyModel,
}

impl BpeNgramBaseline {
    /// Trains the BPE tokenizer (vocabulary `vocab_size`) and the n-gram model on `texts`.
    pub fn train(texts: &[&str], vocab_size: usize) -> Result<Self, String> {
        let mut trainer = BpeTrainerBuilder::new()
            .show_progress(false)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(ByteLevel::alphabet())
            .build();

        let mut builder = TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(None::<NFC>)
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(None::<ByteLevel>)
            .with_decoder(Some(ByteLevelDecoder::default()))
            .build()
            .map_err(|e| e.to_string())?;

        builder
            .train(&mut trainer, texts.iter().copied())
            .map_err(|e| e.to_string())?;
        let tokenizer: Tokenizer = builder.into();

        let mut me = Self {
            tokenizer,
            model: SectorKneserNeyModel::new(0),
        };
        let encoded_all: Vec<Vec<String>> = texts
            .par_iter()
            .map(|&t| me.encode_pieces(t))
            .collect::<Result<Vec<_>, _>>()?;
        let mut frequencies: HashMap<String, u64> = HashMap::new();
        for pieces in &encoded_all {
            for piece in pieces { *frequencies.entry(piece.clone()).or_default() += 1; }
        }
        me.model.set_singleton_tokens(frequencies.iter().filter_map(|(piece, &count)| (count == 1).then_some(piece.clone())).collect());
        for pieces in &encoded_all {
            let refs: Vec<&str> = pieces.iter().map(|s| s.as_str()).collect();
            me.model.train(&refs);
        }
        me.model.recompute_continuation_counts();
        Ok(me)
    }

    /// BPE pieces of `text` (token strings in the tokenizer's byte-level alphabet).
    pub fn encode_pieces(&self, text: &str) -> Result<Vec<String>, String> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        Ok(enc.get_tokens().to_vec())
    }

    /// BPE pieces plus their source character offsets, for diagnostic attribution.
    pub fn encode_pieces_with_offsets(&self, text: &str) -> Result<Vec<(String, usize, usize)>, String> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        let char_count = text.chars().count();
        let max_end = enc.get_offsets().iter().map(|&(_, e)| e).max().unwrap_or(0);
        let is_char_offset = max_end <= char_count;
        let to_char = |byte_pos: usize| text.get(..byte_pos).map_or(char_count, |s| s.chars().count());
        Ok(enc.get_tokens().iter().zip(enc.get_offsets()).map(|(piece, &(start, end))| {
            let (a, b) = if is_char_offset { (start, end) } else { (to_char(start), to_char(end)) };
            (piece.clone(), a, b)
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_is_trained_and_model_learns_repetition() {
        let line = "for i in range(10):\n    print(i)\n";
        let texts: Vec<String> = (0..50).map(|_| line.repeat(4)).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let b = BpeNgramBaseline::train(&refs, 300).unwrap();

        // A trained tokenizer merges frequent byte pairs: fewer pieces than bytes.
        let pieces = b.encode_pieces(line).unwrap();
        assert!(!pieces.is_empty());
        assert!(pieces.len() < line.len(), "no merges learned: {} pieces for {} bytes", pieces.len(), line.len());

        // After the context of a repeated line, the real next piece must be top-1.
        let hist: Vec<&str> = pieces[..4].iter().map(|s| s.as_str()).collect();
        let top = b.model.top_k(&hist, 1, None, 0.0);
        assert_eq!(top[0].0, pieces[4]);
    }
}
