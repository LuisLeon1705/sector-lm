//! Character-level spelling model for words missing from a sector's vocabulary.
//!
//! Interpolated Witten-Bell trigram over characters, backing off to a uniform
//! distribution over all Unicode scalar values plus an end-of-word symbol. The
//! probability of a string is the product over its characters and the end
//! symbol, so the model is a proper distribution over strings.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Number of distinct symbols that can be predicted: every `char` plus END.
const ALPHABET: u64 = 0x11_0000 + 1;
const END: u32 = 0x11_0000;
const BOS: u32 = 0x11_0001;
/// Training uses at most this many leading characters of each word.
pub const MAX_TRAIN_CHARS: usize = 128;
const MAX_CTX: usize = 2;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharNgram {
    /// Per context length k (0..=2): (context, symbol) -> count
    next: [HashMap<u64, u32>; MAX_CTX + 1],
    /// Per context length k: context -> (total count, distinct symbols)
    ctx: [HashMap<u64, (u32, u32)>; MAX_CTX + 1],
}

#[inline]
fn ctx_key(syms: &[u32]) -> u64 {
    syms.iter().fold(0u64, |acc, &s| (acc << 21) | s as u64)
}

impl CharNgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one word (a vocabulary type) to the statistics.
    pub fn train_word(&mut self, word: &str) {
        self.train_word_weighted(word, 1);
    }

    pub fn train_word_weighted(&mut self, word: &str, weight: u32) {
        if weight == 0 { return; }
        let mut syms: Vec<u32> = vec![BOS; MAX_CTX];
        let mut truncated = false;
        for (i, c) in word.chars().enumerate() {
            if i >= MAX_TRAIN_CHARS {
                truncated = true;
                break;
            }
            syms.push(c as u32);
        }
        if !truncated {
            syms.push(END);
        }
        for j in MAX_CTX..syms.len() {
            let sym = syms[j] as u64;
            for k in 0..=MAX_CTX {
                let ck = ctx_key(&syms[j - k..j]);
                let nk = (ck << 21) | sym;
                let stat = self.ctx[k].entry(ck).or_insert((0, 0));
                let cnt = self.next[k].entry(nk).or_insert(0);
                if *cnt == 0 {
                    stat.1 += 1;
                }
                stat.0 += weight;
                *cnt += weight;
            }
        }
    }

    /// P(sym | history) for a history of at least MAX_CTX symbols (BOS-padded).
    fn p_sym(&self, hist: &[u32], sym: u32) -> f64 {
        let mut p = 1.0 / ALPHABET as f64;
        for k in 0..=MAX_CTX {
            let ck = ctx_key(&hist[hist.len() - k..]);
            if let Some(&(total, distinct)) = self.ctx[k].get(&ck) {
                let c = self.next[k].get(&((ck << 21) | sym as u64)).copied().unwrap_or(0) as f64;
                let d = distinct as f64;
                p = (c + d * p) / (total as f64 + d);
            }
        }
        p
    }

    /// Natural log of the probability of spelling `word`, including the end symbol.
    pub fn log_prob(&self, word: &str) -> f64 {
        let mut hist: Vec<u32> = vec![BOS; MAX_CTX];
        let mut lp = 0.0;
        for c in word.chars() {
            lp += self.p_sym(&hist, c as u32).ln();
            hist.push(c as u32);
        }
        lp += self.p_sym(&hist, END).ln();
        lp
    }

    /// Probability of the next symbol being `sym` (END = `END_SYMBOL`) after `prefix`;
    /// exposed so tests can verify normalization over the alphabet.
    pub fn next_symbol_prob(&self, prefix: &str, sym: u32) -> f64 {
        let mut hist: Vec<u32> = vec![BOS; MAX_CTX];
        hist.extend(prefix.chars().map(|c| c as u32));
        self.p_sym(&hist, sym)
    }

    pub fn num_entries(&self) -> usize {
        self.next.iter().map(|m| m.len()).sum()
    }

    /// Symbols that have been observed after any context (for normalization checks).
    pub fn observed_symbols(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.next[0].keys().map(|&k| (k & ((1 << 21) - 1)) as u32).collect();
        v.sort_unstable();
        v
    }
}

pub const END_SYMBOL: u32 = END;
pub const ALPHABET_SIZE: u64 = ALPHABET;

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> CharNgram {
        let mut m = CharNgram::new();
        for w in ["self", "selfish", "shelf", "name", "names", "namespace", "x", "xy"] {
            m.train_word(w);
        }
        m
    }

    #[test]
    fn next_symbol_distribution_sums_to_one() {
        let m = trained();
        let seen: std::collections::HashSet<u32> = m.observed_symbols().into_iter().collect();
        // Seen symbols explicitly; every unseen symbol shares the same probability
        // within a context, so the remainder is (ALPHABET - seen) times that value.
        let unseen_sym = 0x1F600u32; // a char never trained on
        assert!(!seen.contains(&unseen_sym));
        for prefix in ["", "s", "se", "na", "zz", "name"] {
            let mut total: f64 = seen.iter().map(|&s| m.next_symbol_prob(prefix, s)).sum();
            total += (ALPHABET_SIZE - seen.len() as u64) as f64 * m.next_symbol_prob(prefix, unseen_sym);
            assert!((total - 1.0).abs() < 1e-9, "prefix {prefix:?}: sums to {total}");
        }
    }

    #[test]
    fn seen_words_are_likelier_than_scrambles_and_longer_words_cost_more() {
        let m = trained();
        assert!(m.log_prob("names") > m.log_prob("smane"));
        assert!(m.log_prob("name") > m.log_prob("namenamename"));
    }

    #[test]
    fn frequency_weighting_increases_probability_of_frequent_training_words() {
        let mut types_only = CharNgram::new();
        let mut weighted = CharNgram::new();
        for word in ["common", "rare"] { types_only.train_word(word); weighted.train_word_weighted(word, if word == "common" { 100 } else { 1 }); }
        assert!(weighted.log_prob("common") > types_only.log_prob("common"));
    }
}
