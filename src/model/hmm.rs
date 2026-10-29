use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::sector::NUM_SECTORS;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorHmm {
    // Initial state probabilities: log P(S_0 = s)
    pub log_initial: [f64; NUM_SECTORS],
    // Transition probabilities: log P(S_t = j | S_{t-1} = i)
    pub log_transitions: [[f64; NUM_SECTORS]; NUM_SECTORS],
    // Emission counts: sector -> text -> count
    pub emission_counts: Vec<HashMap<String, u32>>,
    // Total emission count per sector
    pub sector_token_totals: [u64; NUM_SECTORS],
    // Vocabulary size across all sectors
    pub vocab_size: usize,
    // Smoothing parameter (Laplace / Add-alpha)
    pub alpha: f64,
    // Active number of sectors (<= NUM_SECTORS)
    pub num_sectors: usize,
}

impl Default for SectorHmm {
    fn default() -> Self {
        Self::new()
    }
}

impl SectorHmm {
    pub fn new() -> Self {
        Self::with_num_sectors(NUM_SECTORS)
    }

    pub fn with_num_sectors(num_sectors: usize) -> Self {
        let num_sectors = num_sectors.clamp(1, NUM_SECTORS);
        let mut emission_counts = Vec::with_capacity(NUM_SECTORS);
        for _ in 0..NUM_SECTORS {
            emission_counts.push(HashMap::new());
        }

        // Initialize uniform probabilities
        let init_prob = (1.0 / num_sectors as f64).ln();
        let log_initial = [init_prob; NUM_SECTORS];
        let log_transitions = [[init_prob; NUM_SECTORS]; NUM_SECTORS];

        Self {
            log_initial,
            log_transitions,
            emission_counts,
            sector_token_totals: [0; NUM_SECTORS],
            vocab_size: 0,
            alpha: 0.1,
            num_sectors,
        }
    }

    /// Fits the HMM parameters on sequences of labeled (sector, text) pairs.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(&mut self, sequences: &[Vec<(u8, String)>]) {
        let n_sec = self.num_sectors;
        let mut initial_counts = [0.0; NUM_SECTORS];
        let mut transition_counts = [[0.0; NUM_SECTORS]; NUM_SECTORS];
        let mut global_vocab = std::collections::HashSet::new();

        for seq in sequences {
            if seq.is_empty() {
                continue;
            }
            let first_sector = seq[0].0 as usize;
            if first_sector < n_sec {
                initial_counts[first_sector] += 1.0;
            }

            for i in 0..seq.len() {
                let (sec, text) = &seq[i];
                let sec_idx = *sec as usize;
                if sec_idx >= n_sec {
                    continue;
                }

                global_vocab.insert(text.clone());
                *self.emission_counts[sec_idx]
                    .entry(text.clone())
                    .or_insert(0) += 1;
                self.sector_token_totals[sec_idx] += 1;

                if i + 1 < seq.len() {
                    let next_sec = seq[i + 1].0 as usize;
                    if next_sec < n_sec {
                        transition_counts[sec_idx][next_sec] += 1.0;
                    }
                }
            }
        }

        self.vocab_size = global_vocab.len().max(1);

        // Compute smoothed initial probabilities
        let total_init: f64 = initial_counts[..n_sec].iter().sum::<f64>() + (n_sec as f64 * self.alpha);
        for i in 0..n_sec {
            self.log_initial[i] = ((initial_counts[i] + self.alpha) / total_init).ln();
        }

        // Compute smoothed transition probabilities
        for i in 0..n_sec {
            let row_sum: f64 = transition_counts[i][..n_sec].iter().sum::<f64>() + (n_sec as f64 * self.alpha);
            for j in 0..n_sec {
                self.log_transitions[i][j] = ((transition_counts[i][j] + self.alpha) / row_sum).ln();
            }
        }
    }

    /// log P(S_t = cur | S_{t-1} = prev): the price of choosing sector cur after prev.
    pub fn log_transition(&self, prev: u8, cur: u8) -> f64 {
        self.log_transitions[prev as usize][cur as usize]
    }

    /// Computes smoothed log emission probability log P(token | sector)
    pub fn log_emission(&self, sector: usize, token: &str) -> f64 {
        if sector >= NUM_SECTORS {
            return -100.0;
        }

        let count = self.emission_counts[sector].get(token).copied().unwrap_or(0) as f64;
        let total = self.sector_token_totals[sector] as f64;
        let denom = total + self.alpha * (self.vocab_size as f64 + 1.0);

        ((count + self.alpha) / denom).ln()
    }

    /// Viterbi decoding algorithm:
    /// Given observed tokens w_0..w_{T-1}, finds argmax_S P(S_0..S_{T-1} | w_0..w_{T-1})
    #[allow(clippy::needless_range_loop)]
    pub fn decode_viterbi(&self, tokens: &[&str]) -> Vec<u8> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let n_sec = self.num_sectors;
        let t_len = tokens.len();
        // delta[t][j] = max log prob of state j at step t
        let mut delta = vec![[f64::NEG_INFINITY; NUM_SECTORS]; t_len];
        // psi[t][j] = backpointer to best state i at step t-1
        let mut psi = vec![[0usize; NUM_SECTORS]; t_len];

        // Step 0: Initialization
        for j in 0..n_sec {
            delta[0][j] = self.log_initial[j] + self.log_emission(j, tokens[0]);
        }

        // Step 1..T-1: Recursion
        for t in 1..t_len {
            for j in 0..n_sec {
                let mut best_val = f64::NEG_INFINITY;
                let mut best_i = 0;
                let emit = self.log_emission(j, tokens[t]);

                for i in 0..n_sec {
                    let score = delta[t - 1][i] + self.log_transitions[i][j];
                    if score > best_val {
                        best_val = score;
                        best_i = i;
                    }
                }
                delta[t][j] = best_val + emit;
                psi[t][j] = best_i;
            }
        }

        // Termination
        let mut best_last_state = 0;
        let mut max_final = f64::NEG_INFINITY;
        for j in 0..n_sec {
            if delta[t_len - 1][j] > max_final {
                max_final = delta[t_len - 1][j];
                best_last_state = j;
            }
        }

        // Backtracking
        let mut path = vec![0u8; t_len];
        path[t_len - 1] = best_last_state as u8;
        let mut cur = best_last_state;

        for t in (1..t_len).rev() {
            cur = psi[t][cur];
            path[t - 1] = cur as u8;
        }

        path
    }

    /// Evaluates precision/accuracy of recovering the correct sector compared to ground-truth labels.
    pub fn evaluate_accuracy(&self, sequences: &[Vec<(u8, String)>]) -> (f64, usize, usize) {
        let mut correct = 0;
        let mut total = 0;

        for seq in sequences {
            let tokens: Vec<&str> = seq.iter().map(|(_, t)| t.as_str()).collect();
            let predicted = self.decode_viterbi(&tokens);
            for (pred, (truth, _)) in predicted.iter().zip(seq.iter()) {
                if *pred == *truth {
                    correct += 1;
                }
                total += 1;
            }
        }

        let acc = if total == 0 { 0.0 } else { correct as f64 / total as f64 };
        (acc, correct, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::SectorKind;

    #[test]
    fn test_viterbi_disambiguation_by_context() {
        let mut hmm = SectorHmm::new();

        let kw = SectorKind::Keyword.id();
        let def_name = SectorKind::DefinedName.id();
        let used_name = SectorKind::UsedName.id();
        let delim = SectorKind::Delimiter.id();

        // Train on typical Python sequences:
        // "def" (kw) -> "func" (def_name) -> ":" (delim)
        // "func" (used_name) -> "(" (delim)
        // "in" as keyword in "for x in y" vs "in" as operator in "if x in y"
        let mut sequences = Vec::new();
        for _ in 0..50 {
            sequences.push(vec![
                (kw, "def".to_string()),
                (def_name, "foo".to_string()),
                (delim, "(".to_string()),
                (delim, ")".to_string()),
                (delim, ":".to_string()),
            ]);
            sequences.push(vec![
                (used_name, "foo".to_string()),
                (delim, "(".to_string()),
                (delim, ")".to_string()),
            ]);
            // Ambiguous token "foo": appears as DefinedName right after "def",
            // and as UsedName when alone or followed by "("
        }

        hmm.fit(&sequences);

        // Context 1: After "def", "foo" MUST be classified as DefinedName
        let ctx1 = vec!["def", "foo", "(", ")", ":"];
        let path1 = hmm.decode_viterbi(&ctx1);
        assert_eq!(path1[0], kw, "def must be Keyword");
        assert_eq!(path1[1], def_name, "foo after def must be DefinedName");

        // Context 2: "foo" followed by "(" as an expression must be classified as UsedName
        let ctx2 = vec!["foo", "(", ")"];
        let path2 = hmm.decode_viterbi(&ctx2);
        assert_eq!(path2[0], used_name, "standalone foo must be UsedName");
    }
}
