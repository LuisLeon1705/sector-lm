use std::io::IsTerminal;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::hmm::SectorHmm;
use crate::model::history::{HistoryMode, history_context};
use crate::model::ngram::{SectorKneserNeyModel, SpellCache};
use crate::sector::parser::ParsedToken;
use crate::sector::{SectorKind, NUM_SECTORS};
use crate::storage::cache::SectorLruCache;
use crate::symbol::table::SymbolTable;

/// Progress shared between threads. On a terminal it draws an `indicatif` bar;
/// otherwise it prints one line per 10 % (percentage, done/total, rate, ETA).
pub struct ProgressTracker {
    label: String,
    unit: &'static str,
    total: u64,
    done: AtomicU64,
    next_decile: AtomicU64,
    start: Instant,
    bar: Option<ProgressBar>,
}

impl ProgressTracker {
    pub fn new(label: &str, unit: &'static str, total: u64) -> Self {
        let bar = if std::io::stdout().is_terminal() {
            let pb = ProgressBar::new(total);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template(&format!(
                        "{{spinner:.green}} [{label}] [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} {unit} ({{percent}}%) {{per_sec}} ETA {{eta}}"
                    ))
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };
        Self {
            label: label.to_string(),
            unit,
            total,
            done: AtomicU64::new(0),
            next_decile: AtomicU64::new(1),
            start: Instant::now(),
            bar,
        }
    }

    pub fn inc(&self, n: u64) {
        let done = self.done.fetch_add(n, AtomicOrdering::Relaxed) + n;
        if let Some(b) = &self.bar {
            b.inc(n);
            return;
        }
        if self.total == 0 {
            return;
        }
        let decile = (done.min(self.total) * 10) / self.total;
        loop {
            let next = self.next_decile.load(AtomicOrdering::Relaxed);
            if decile < next {
                break;
            }
            // Only the thread that advances the counter prints, so each decile prints once.
            if self
                .next_decile
                .compare_exchange(next, decile + 1, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
                .is_ok()
            {
                let el = self.start.elapsed().as_secs_f64();
                let rate = done as f64 / el.max(1e-9);
                let eta = (self.total.saturating_sub(done)) as f64 / rate.max(1e-9);
                println!(
                    "[{}] {:>3}% | {}/{} {} | {:.0} {}/s | elapsed {:.0}s | ETA {:.0}s",
                    self.label,
                    decile * 10,
                    done,
                    self.total,
                    self.unit,
                    rate,
                    self.unit,
                    el,
                    eta
                );
                break;
            }
        }
    }

    pub fn println(&self, msg: &str) {
        match &self.bar {
            Some(b) => b.println(msg),
            None => println!("{}", msg),
        }
    }

    pub fn finish(&self) {
        if let Some(b) = &self.bar {
            b.finish_and_clear();
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvalOptions {
    /// Correction 1: score each token as P(sector | prev_sector) * P(token | sector, context).
    pub price_sector: bool,
    /// Correction 3: score unseen tokens as P(<unk> | context) * P(character spelling).
    pub use_spelling: bool,
    /// Keep file-level domain sectors fixed instead of decoding them with the syntax HMM.
    pub fixed_sectors: bool,
    pub history_mode: HistoryMode,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            price_sector: true,
            use_spelling: true,
            fixed_sectors: false,
            history_mode: HistoryMode::Sector,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub top1_accuracy: f64,
    pub top5_accuracy: f64,
    pub perplexity_per_char: f64,
    pub memory_mb: f64,
    pub tokens_per_sec: f64,
    pub sectors_loaded_per_1000_tokens: Option<f64>,
    pub bytes_loaded_per_1000_tokens: Option<f64>,
    pub symbol_resolution_accuracy: Option<f64>,
    pub sector_disambiguation_accuracy: Option<f64>,
    pub exact_total_chars: u64,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct EvaluatorAccumulator {
    pub total_predictions: u64,
    pub top1_hits: u64,
    pub top5_hits: u64,
    pub total_chars: u64,
    pub total_log_prob: f64,
    pub elapsed_secs: f64,

    // Specific to defined name resolution
    pub defined_name_predictions: u64,
    pub defined_name_top1_hits: u64,
    pub defined_name_top5_hits: u64,

    // Diagnostics per sector (index = sector id): nats spent, tokens scored, and
    // tokens whose text was outside the sector vocabulary.
    pub per_sector_nats: [f64; NUM_SECTORS],
    pub per_sector_tokens: [u64; NUM_SECTORS],
    pub per_sector_oov: [u64; NUM_SECTORS],
    // Nats spent on choosing the sector (0 when the sector is not priced)
    pub price_nats: f64,
    /// Character-loss breakdown: code, comments, strings, format.
    pub category_nats: [f64; 4],
    pub category_chars: [u64; 4],
    pub unseen_spelling_nats: f64,
    pub unseen_spelling_chars: u64,
}

pub fn text_category(token: &ParsedToken) -> usize {
    if token.sector == SectorKind::Format || token.rule == 'F' { return 3; }
    if token.sector == SectorKind::Comment || token.rule == 'M' || (token.sector == SectorKind::TextPiece && token.condition == 'M') { return 1; }
    if token.sector == SectorKind::StringLiteral || token.rule == 'S' || (token.sector == SectorKind::TextPiece && token.condition == 'S') { return 2; }
    0
}

impl EvaluatorAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&mut self, other: &EvaluatorAccumulator) {
        self.total_predictions += other.total_predictions;
        self.top1_hits += other.top1_hits;
        self.top5_hits += other.top5_hits;
        self.total_chars += other.total_chars;
        self.total_log_prob += other.total_log_prob;
        self.defined_name_predictions += other.defined_name_predictions;
        self.defined_name_top1_hits += other.defined_name_top1_hits;
        self.defined_name_top5_hits += other.defined_name_top5_hits;
        for s in 0..NUM_SECTORS {
            self.per_sector_nats[s] += other.per_sector_nats[s];
            self.per_sector_tokens[s] += other.per_sector_tokens[s];
            self.per_sector_oov[s] += other.per_sector_oov[s];
        }
        self.price_nats += other.price_nats;
        for c in 0..4 { self.category_nats[c] += other.category_nats[c]; self.category_chars[c] += other.category_chars[c]; }
        self.unseen_spelling_nats += other.unseen_spelling_nats;
        self.unseen_spelling_chars += other.unseen_spelling_chars;
    }

    pub fn record_prediction(
        &mut self,
        target: &str,
        candidates_top_k: &[(String, f64)],
        target_prob: f64,
        is_defined_name: bool,
    ) {
        let safe_prob = target_prob.max(1e-12);
        self.record_prediction_log(target, candidates_top_k, safe_prob.ln(), is_defined_name);
    }

    /// Records a prediction given the natural log of the target's probability
    /// (no clamping; used where probabilities can be extremely small).
    pub fn record_prediction_log(
        &mut self,
        target: &str,
        candidates_top_k: &[(String, f64)],
        target_log_prob: f64,
        is_defined_name: bool,
    ) {
        self.total_predictions += 1;
        self.total_log_prob += target_log_prob;

        let in_top1 = candidates_top_k.first().is_some_and(|(c, _)| c == target);
        let in_top5 = candidates_top_k.iter().take(5).any(|(c, _)| c == target);

        if in_top1 {
            self.top1_hits += 1;
        }
        if in_top5 {
            self.top5_hits += 1;
        }

        if is_defined_name {
            self.defined_name_predictions += 1;
            if in_top1 {
                self.defined_name_top1_hits += 1;
            }
            if in_top5 {
                self.defined_name_top5_hits += 1;
            }
        }
    }

    pub fn perplexity_per_char(&self) -> f64 {
        if self.total_chars > 0 {
            (-self.total_log_prob / self.total_chars as f64).exp()
        } else {
            1.0
        }
    }

    pub fn compute_metrics(
        &self,
        model_name: &str,
        memory_mb: f64,
        sectors_per_1000: Option<f64>,
        bytes_per_1000: Option<f64>,
        sector_disambiguation_acc: Option<f64>,
    ) -> ModelMetrics {
        let top1_accuracy = if self.total_predictions > 0 {
            (self.top1_hits as f64) / (self.total_predictions as f64)
        } else {
            0.0
        };

        let top5_accuracy = if self.total_predictions > 0 {
            (self.top5_hits as f64) / (self.total_predictions as f64)
        } else {
            0.0
        };

        let perplexity_per_char = self.perplexity_per_char();

        let tokens_per_sec = if self.elapsed_secs > 0.0 {
            (self.total_predictions as f64) / self.elapsed_secs
        } else {
            0.0
        };

        let symbol_resolution_accuracy = if self.defined_name_predictions > 0 {
            Some((self.defined_name_top1_hits as f64) / (self.defined_name_predictions as f64))
        } else {
            None
        };

        ModelMetrics {
            model_name: model_name.to_string(),
            top1_accuracy,
            top5_accuracy,
            perplexity_per_char,
            memory_mb,
            tokens_per_sec,
            sectors_loaded_per_1000_tokens: sectors_per_1000,
            bytes_loaded_per_1000_tokens: bytes_per_1000,
            symbol_resolution_accuracy,
            sector_disambiguation_accuracy: sector_disambiguation_acc,
            exact_total_chars: self.total_chars,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SingleFileEvalResult {
    pub file_index: usize,
    pub accum_with_copy: EvaluatorAccumulator,
    pub accum_no_copy: EvaluatorAccumulator,
    pub accessed_sectors: Vec<u8>,
    pub viterbi_correct: usize,
    pub viterbi_total: usize,
    pub exact_char_count: usize,
    pub token_count: usize,
    pub elapsed_secs: f64,
}

/// Evaluates a single file independently using default corrected options.
pub fn evaluate_single_file_sector_lm(
    file_index: usize,
    content: &str,
    tokens: &[ParsedToken],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    pb: Option<&ProgressTracker>,
) -> SingleFileEvalResult {
    evaluate_single_file_sector_lm_opts(
        file_index,
        content,
        tokens,
        hmm,
        sector_models,
        copy_weight,
        EvalOptions::default(),
        pb,
    )
}

/// Evaluates a single file independently with explicit `EvalOptions`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_single_file_sector_lm_opts(
    file_index: usize,
    content: &str,
    tokens: &[ParsedToken],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    opts: EvalOptions,
    pb: Option<&ProgressTracker>,
) -> SingleFileEvalResult {
    let start_t = Instant::now();
    let mut accum_with_copy = EvaluatorAccumulator::new();
    let mut accum_no_copy = EvaluatorAccumulator::new();
    let mut accessed_sectors = Vec::with_capacity(tokens.len());
    let mut sym_table = SymbolTable::new();
    let spell_caches: Vec<SpellCache> = (0..sector_models.len()).map(|_| SpellCache::default()).collect();

    let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
    for tok in tokens {
        let c = text_category(tok);
        let chars = tok.text.chars().count() as u64;
        accum_with_copy.category_chars[c] += chars;
        accum_no_copy.category_chars[c] += chars;
    }

    // 1. Sector Disambiguation (HMM Viterbi)
    let decoded = if opts.fixed_sectors { tokens.iter().map(|t| t.sector.id()).collect() } else { hmm.decode_viterbi(&texts) };
    let mut viterbi_correct = 0;
    let mut viterbi_total = 0;
    for (pred_sec, tok) in decoded.iter().zip(tokens.iter()) {
        if *pred_sec == tok.sector.id() {
            viterbi_correct += 1;
        }
        viterbi_total += 1;
    }

    // 2. Next-token prediction
    let mut batch_counter = 0;
    for i in 0..tokens.len() {
        let tok = &tokens[i];

        if tok.is_definition {
            sym_table.define(&tok.text, tok.sector.id(), tok.condition, tok.byte_range.0);
        }

        if i >= 1 {
            let target = tok.text.as_str();
            let context = history_context(tokens, i, opts.history_mode);
            let history: Vec<&str> = context.iter().map(String::as_str).collect();

            let prev_sec = tokens[i - 1].sector.id();
            let sec_id = tok.sector.id();
            accessed_sectors.push(sec_id);

            let s_idx = sec_id as usize;
            let sec_model = &sector_models[s_idx];
            // A token is a used identifier reference if its sector is UsedName or its syntactic rule is 'R'
            let is_used_name = tok.sector == SectorKind::UsedName || tok.rule == 'R';

            let symbols = if is_used_name { Some(&sym_table) } else { None };
            let spell_opt = if opts.use_spelling {
                Some(&spell_caches[s_idx])
            } else {
                None
            };

            let (top5_copy, top5_no_copy) =
                sec_model.top_k_dual_opts(&history, 5, symbols, copy_weight, spell_opt);

            let history_ids: Vec<u32> = history.iter().map(|&h| sec_model.get_token_id_or_unk(h)).collect();
            let is_oov = sec_model.get_token_id(target).is_none();

            let scoring_spell = if opts.use_spelling { spell_opt } else { None };
            let base_lp = sec_model.log_base_prob_text(target, &history_ids, scoring_spell);
            let copy_lp = sec_model.log_prob_token_opts(target, &history, symbols, copy_weight, scoring_spell);
            let price_lp = if opts.price_sector {
                hmm.log_transition(prev_sec, sec_id)
            } else {
                0.0
            };

            accum_with_copy.record_prediction_log(target, &top5_copy, price_lp + copy_lp, is_used_name);
            accum_no_copy.record_prediction_log(target, &top5_no_copy, price_lp + base_lp, is_used_name);

            let category = text_category(tok);
            accum_with_copy.category_nats[category] += -(price_lp + copy_lp);
            accum_no_copy.category_nats[category] += -(price_lp + base_lp);
            if is_oov && opts.use_spelling {
                if let Some(spelling) = &sec_model.spelling {
                    let nats = -spelling.log_prob(target);
                    let chars = target.chars().count() as u64;
                    accum_with_copy.unseen_spelling_nats += nats;
                    accum_with_copy.unseen_spelling_chars += chars;
                    accum_no_copy.unseen_spelling_nats += nats;
                    accum_no_copy.unseen_spelling_chars += chars;
                }
            }

            if s_idx < NUM_SECTORS {
                accum_with_copy.per_sector_nats[s_idx] += -copy_lp;
                accum_with_copy.per_sector_tokens[s_idx] += 1;
                if is_oov {
                    accum_with_copy.per_sector_oov[s_idx] += 1;
                }
                accum_no_copy.per_sector_nats[s_idx] += -base_lp;
                accum_no_copy.per_sector_tokens[s_idx] += 1;
                if is_oov {
                    accum_no_copy.per_sector_oov[s_idx] += 1;
                }
            }
            accum_with_copy.price_nats += -price_lp;
            accum_no_copy.price_nats += -price_lp;

            batch_counter += 1;
            if batch_counter >= 256 {
                if let Some(p) = pb {
                    p.inc(batch_counter as u64);
                }
                batch_counter = 0;
            }
        }
    }

    if batch_counter > 0 {
        if let Some(p) = pb {
            p.inc(batch_counter as u64);
        }
    }

    let exact_char_count = content.chars().count();
    accum_with_copy.total_chars = exact_char_count as u64;
    accum_no_copy.total_chars = exact_char_count as u64;

    let elapsed_secs = start_t.elapsed().as_secs_f64();

    if let Some(p) = pb {
        let mut line = format!(
            "[file] idx={} tokens={} chars={} time={:.2}s",
            file_index,
            tokens.len(),
            exact_char_count,
            elapsed_secs
        );
        if elapsed_secs > 60.0 {
            line.push_str("  WARNING: exceeds 60 s");
        }
        p.println(&line);
    }

    SingleFileEvalResult {
        file_index,
        accum_with_copy,
        accum_no_copy,
        accessed_sectors,
        viterbi_correct,
        viterbi_total,
        exact_char_count,
        token_count: tokens.len(),
        elapsed_secs,
    }
}

/// Sequential evaluation over a slice of files
pub fn evaluate_dataset_sector_lm_sequential(
    files: &[(String, Vec<ParsedToken>)],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    cache: &mut SectorLruCache,
) -> (EvaluatorAccumulator, EvaluatorAccumulator, usize, usize) {
    evaluate_dataset_sector_lm_sequential_opts(
        files,
        hmm,
        sector_models,
        copy_weight,
        EvalOptions::default(),
        cache,
    )
}

pub fn evaluate_dataset_sector_lm_sequential_opts(
    files: &[(String, Vec<ParsedToken>)],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    opts: EvalOptions,
    cache: &mut SectorLruCache,
) -> (EvaluatorAccumulator, EvaluatorAccumulator, usize, usize) {
    let mut total_with_copy = EvaluatorAccumulator::new();
    let mut total_no_copy = EvaluatorAccumulator::new();
    let mut total_viterbi_correct = 0;
    let mut total_viterbi_total = 0;

    for (idx, (content, tokens)) in files.iter().enumerate() {
        let res = evaluate_single_file_sector_lm_opts(
            idx,
            content,
            tokens,
            hmm,
            sector_models,
            copy_weight,
            opts,
            None,
        );

        for &sec in &res.accessed_sectors {
            let _ = cache.record_access(sec);
        }
        cache.record_predicted_tokens(res.accum_with_copy.total_predictions);

        total_with_copy.merge(&res.accum_with_copy);
        total_no_copy.merge(&res.accum_no_copy);
        total_viterbi_correct += res.viterbi_correct;
        total_viterbi_total += res.viterbi_total;
    }

    (
        total_with_copy,
        total_no_copy,
        total_viterbi_correct,
        total_viterbi_total,
    )
}

/// Parallel evaluation across CPU cores using Rayon.
/// Results and LRU cache playback are guaranteed 100% identical to sequential.
pub fn evaluate_dataset_sector_lm_parallel(
    files: &[(String, Vec<ParsedToken>)],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    cache: &mut SectorLruCache,
    pb: Option<&ProgressTracker>,
) -> (
    EvaluatorAccumulator,
    EvaluatorAccumulator,
    usize,
    usize,
    Vec<SingleFileEvalResult>,
) {
    evaluate_dataset_sector_lm_parallel_opts(
        files,
        hmm,
        sector_models,
        copy_weight,
        EvalOptions::default(),
        cache,
        pb,
    )
}

pub fn evaluate_dataset_sector_lm_parallel_opts(
    files: &[(String, Vec<ParsedToken>)],
    hmm: &SectorHmm,
    sector_models: &[SectorKneserNeyModel],
    copy_weight: f64,
    opts: EvalOptions,
    cache: &mut SectorLruCache,
    pb: Option<&ProgressTracker>,
) -> (
    EvaluatorAccumulator,
    EvaluatorAccumulator,
    usize,
    usize,
    Vec<SingleFileEvalResult>,
) {
    let mut indexed_files: Vec<(usize, &(String, Vec<ParsedToken>))> =
        files.iter().enumerate().collect();
    indexed_files.sort_by_key(|b| std::cmp::Reverse(b.1.1.len()));

    let mut results: Vec<SingleFileEvalResult> = indexed_files
        .par_iter()
        .map(|&(orig_idx, (content, tokens))| {
            evaluate_single_file_sector_lm_opts(
                orig_idx,
                content,
                tokens,
                hmm,
                sector_models,
                copy_weight,
                opts,
                pb,
            )
        })
        .collect();

    results.sort_by_key(|r| r.file_index);

    let mut total_with_copy = EvaluatorAccumulator::new();
    let mut total_no_copy = EvaluatorAccumulator::new();
    let mut total_viterbi_correct = 0;
    let mut total_viterbi_total = 0;

    for res in &results {
        for &sec in &res.accessed_sectors {
            let _ = cache.record_access(sec);
        }
        cache.record_predicted_tokens(res.accum_with_copy.total_predictions);

        total_with_copy.merge(&res.accum_with_copy);
        total_no_copy.merge(&res.accum_no_copy);
        total_viterbi_correct += res.viterbi_correct;
        total_viterbi_total += res.viterbi_total;
    }

    (
        total_with_copy,
        total_no_copy,
        total_viterbi_correct,
        total_viterbi_total,
        results,
    )
}
