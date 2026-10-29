use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::model::spelling::CharNgram;
use crate::symbol::table::SymbolTable;
use crate::storage::sector_file::SectorNgramCounts;

pub const UNK_TOKEN: &str = "<unk>";
pub const UNK_ID: u32 = 0;
fn default_true() -> bool { true }

/// Cache of log spelling probabilities by word, used for out-of-vocabulary active symbols.
pub type SpellCache = RefCell<HashMap<String, f64>>;

/// Top-k candidate. Greater = better: higher score, then lexicographically smaller text.
#[derive(Debug, Clone, Copy)]
struct Cand<'a> {
    score: f64,
    text: &'a str,
    #[allow(dead_code)]
    id: u32,
}

impl PartialEq for Cand<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cand<'_> {}
impl PartialOrd for Cand<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.text.cmp(self.text))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorKneserNeyModel {
    pub sector_id: u8,
    pub vocab: Vec<String>,
    pub vocab_indices: HashMap<String, u32>,
    pub unigram_counts: HashMap<u32, u32>,
    pub bigram_counts: HashMap<(u32, u32), u32>,
    pub trigram_counts: HashMap<(u32, u32, u32), u32>,
    pub total_tokens: u64,
    pub discount: f64,
    #[serde(default)]
    pub singleton_tokens: HashSet<String>,
    #[serde(default)]
    pub spelling_counts: HashMap<String, u32>,
    #[serde(default = "default_true")]
    pub frequency_weighted_spelling: bool,
    #[serde(default)]
    pub ngram_vocab_ids: Vec<u32>,

    // Precalculated continuation counts for Kneser-Ney
    pub continuation_counts: HashMap<u32, u32>,
    pub total_continuations: u64,
    pub distinct_followers_1: HashMap<u32, u32>,
    pub distinct_followers_2: HashMap<(u32, u32), u32>,
    // Context totals: sum over followers of the bigram / trigram counts. These are the
    // denominators of the Kneser-Ney estimates, so each conditional distribution sums to 1.
    pub ctx1_total: HashMap<u32, u32>,
    pub ctx2_total: HashMap<(u32, u32), u32>,

    /// Character model for words outside the vocabulary (not stored in the n-gram counts).
    #[serde(skip)]
    pub spelling: Option<CharNgram>,

    // Continuations sorted descending by count for exact early-stopping top-k
    pub bigram_conts_sorted: HashMap<u32, Vec<(u32, u32)>>,
    pub trigram_conts_sorted: HashMap<(u32, u32), Vec<(u32, u32)>>,
    pub unigram_sorted_by_cont: Vec<u32>,
    pub max_p_cont: f64,
}

impl SectorKneserNeyModel {
    pub fn new(sector_id: u8) -> Self {
        let mut vocab = Vec::new();
        let mut vocab_indices = HashMap::new();
        // ID 0 is strictly reserved for <unk>
        vocab.push(UNK_TOKEN.to_string());
        vocab_indices.insert(UNK_TOKEN.to_string(), UNK_ID);

        Self {
            sector_id,
            vocab,
            vocab_indices,
            unigram_counts: HashMap::new(),
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
            total_tokens: 0,
            discount: 0.75,
            singleton_tokens: HashSet::new(),
            spelling_counts: HashMap::new(),
            frequency_weighted_spelling: true,
            ngram_vocab_ids: vec![UNK_ID],
            continuation_counts: HashMap::new(),
            total_continuations: 0,
            distinct_followers_1: HashMap::new(),
            distinct_followers_2: HashMap::new(),
            ctx1_total: HashMap::new(),
            ctx2_total: HashMap::new(),
            spelling: None,
            bigram_conts_sorted: HashMap::new(),
            trigram_conts_sorted: HashMap::new(),
            unigram_sorted_by_cont: Vec::new(),
            max_p_cont: 0.0,
        }
    }

    pub fn get_or_add_token(&mut self, text: &str) -> u32 {
        if let Some(&id) = self.vocab_indices.get(text) {
            id
        } else {
            let id = self.vocab.len() as u32;
            self.vocab.push(text.to_string());
            self.vocab_indices.insert(text.to_string(), id);
            id
        }
    }

    pub fn get_token_id(&self, text: &str) -> Option<u32> {
        if self.singleton_tokens.contains(text) { None } else { self.vocab_indices.get(text).copied() }
    }

    #[inline]
    pub fn get_token_id_or_unk(&self, text: &str) -> u32 {
        self.get_token_id(text).unwrap_or(UNK_ID)
    }

    pub fn get_token_text(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Train on a sequence of tokens belonging to this sector
    pub fn train(&mut self, tokens: &[&str]) {
        if tokens.is_empty() {
            return;
        }

        let token_ids: Vec<u32> = tokens.iter().map(|&t| {
            *self.spelling_counts.entry(t.to_string()).or_insert(0) += 1;
            let id = self.get_or_add_token(t);
            if self.singleton_tokens.contains(t) { UNK_ID } else { id }
        }).collect();

        for i in 0..token_ids.len() {
            let w = token_ids[i];
            *self.unigram_counts.entry(w).or_insert(0) += 1;
            self.total_tokens += 1;

            if i >= 1 {
                let u = token_ids[i - 1];
                *self.bigram_counts.entry((u, w)).or_insert(0) += 1;
            }

            if i >= 2 {
                let u1 = token_ids[i - 2];
                let u2 = token_ids[i - 1];
                *self.trigram_counts.entry((u1, u2, w)).or_insert(0) += 1;
            }
        }
    }

    /// Trains one target with an explicit, already-constructed context (used by mixed history).
    /// Context-only vocabulary items participate in n-grams but not in the predicted vocabulary.
    pub fn train_event(&mut self, history: &[&str], target: &str) {
        *self.spelling_counts.entry(target.to_string()).or_insert(0) += 1;
        let ids: Vec<u32> = history.iter().map(|&h| {
            if self.singleton_tokens.contains(h) { UNK_ID } else { self.get_or_add_token(h) }
        }).collect();
        let target_id = self.get_or_add_token(target);
        let target_id = if self.singleton_tokens.contains(target) { UNK_ID } else { target_id };
        *self.unigram_counts.entry(target_id).or_insert(0) += 1;
        self.total_tokens += 1;
        if !ids.is_empty() {
            *self.bigram_counts.entry((*ids.last().unwrap(), target_id)).or_insert(0) += 1;
        }
        if ids.len() >= 2 {
            let n = ids.len();
            *self.trigram_counts.entry((ids[n - 2], ids[n - 1], target_id)).or_insert(0) += 1;
        }
    }

    pub fn set_singleton_tokens(&mut self, singletons: HashSet<String>) {
        self.singleton_tokens = singletons;
    }

    pub fn singleton_count(&self) -> usize { self.singleton_tokens.len() }
    pub fn set_frequency_weighted_spelling(&mut self, enabled: bool) { self.frequency_weighted_spelling = enabled; }

    /// Trains the character model on the vocabulary types of this sector (each distinct
    /// word once, excluding <unk>), i.e. the same data the n-gram counts came from.
    pub fn train_spelling(&mut self) {
        let mut m = CharNgram::new();
        if self.frequency_weighted_spelling {
            for (w, &count) in &self.spelling_counts { m.train_word_weighted(w, count); }
        } else {
            for w in self.vocab.iter().skip(1) { m.train_word(w); }
        }
        self.spelling = Some(m);
    }

    pub fn recompute_continuation_counts(&mut self) {
        self.continuation_counts.clear();
        self.distinct_followers_1.clear();
        self.distinct_followers_2.clear();
        self.ctx1_total.clear();
        self.ctx2_total.clear();
        self.bigram_conts_sorted.clear();
        self.trigram_conts_sorted.clear();

        for (&(u, w), &cnt) in &self.bigram_counts {
            *self.continuation_counts.entry(w).or_insert(0) += 1;
            *self.distinct_followers_1.entry(u).or_insert(0) += 1;
            *self.ctx1_total.entry(u).or_insert(0) += cnt;
            self.bigram_conts_sorted.entry(u).or_default().push((w, cnt));
        }

        for (&(u1, u2, w), &cnt) in &self.trigram_counts {
            *self.distinct_followers_2.entry((u1, u2)).or_insert(0) += 1;
            *self.ctx2_total.entry((u1, u2)).or_insert(0) += cnt;
            self.trigram_conts_sorted.entry((u1, u2)).or_default().push((w, cnt));
        }

        // Sort bigram continuations descending by frequency
        for list in self.bigram_conts_sorted.values_mut() {
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }

        // Sort trigram continuations descending by frequency
        for list in self.trigram_conts_sorted.values_mut() {
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }

        self.total_continuations = self.continuation_counts.values().map(|&v| v as u64).sum();

        // Precalculate unigrams sorted descending by continuation probability, tie-break by text
        self.ngram_vocab_ids = (0..self.vocab.len() as u32)
            .filter(|&id| id == UNK_ID || self.unigram_counts.get(&id).copied().unwrap_or(0) > 0)
            .collect();
        let mut unigram_ids: Vec<u32> = self.ngram_vocab_ids.clone();
        unigram_ids.sort_by(|&a, &b| {
            let p_b = self.p_continuation(b);
            let p_a = self.p_continuation(a);
            p_b.partial_cmp(&p_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.vocab[a as usize].cmp(&self.vocab[b as usize]))
        });
        self.unigram_sorted_by_cont = unigram_ids;

        self.max_p_cont = self
            .unigram_sorted_by_cont
            .first()
            .map(|&w| self.p_continuation(w))
            .unwrap_or(0.0);

        self.train_spelling();
    }

    /// Computes the unigram continuation probability: P_cont(w)
    #[inline]
    pub fn p_continuation(&self, w: u32) -> f64 {
        let v_size = self.ngram_vocab_ids.len().max(1) as f64;
        if self.total_continuations == 0 {
            return 1.0 / v_size;
        }
        let cont = self.continuation_counts.get(&w).copied().unwrap_or(0) as f64;
        if !self.ngram_vocab_ids.is_empty() && !self.ngram_vocab_ids.contains(&w) { return 0.0; }
        (cont + 0.01) / (self.total_continuations as f64 + 0.01 * v_size)
    }

    /// Bigram Kneser-Ney probability: P_KN(w | u)
    #[inline]
    pub fn p_kn_bigram(&self, u: u32, w: u32) -> f64 {
        let c_u = self.ctx1_total.get(&u).copied().unwrap_or(0) as f64;
        let p_cont = self.p_continuation(w);

        if c_u == 0.0 {
            return p_cont;
        }

        let c_uw = self.bigram_counts.get(&(u, w)).copied().unwrap_or(0) as f64;
        let d = self.discount;
        let highest = (c_uw - d).max(0.0) / c_u;

        let num_followers = self.distinct_followers_1.get(&u).copied().unwrap_or(0) as f64;
        let lambda = (d / c_u) * num_followers;

        highest + lambda * p_cont
    }

    /// Trigram Kneser-Ney probability: P_KN(w | u1, u2)
    #[inline]
    pub fn p_kn_trigram(&self, u1: u32, u2: u32, w: u32) -> f64 {
        let c_u12 = self.ctx2_total.get(&(u1, u2)).copied().unwrap_or(0) as f64;
        let p_lower = self.p_kn_bigram(u2, w);

        if c_u12 == 0.0 {
            return p_lower;
        }

        let c_u12w = self.trigram_counts.get(&(u1, u2, w)).copied().unwrap_or(0) as f64;
        let d = self.discount;
        let highest = (c_u12w - d).max(0.0) / c_u12;

        let num_followers = self.distinct_followers_2.get(&(u1, u2)).copied().unwrap_or(0) as f64;
        let lambda = (d / c_u12) * num_followers;

        highest + lambda * p_lower
    }

    /// Predicts base Kneser-Ney probability (without copy)
    #[inline]
    pub fn predict_base_prob_by_id(&self, w_id: u32, history_ids: &[u32]) -> f64 {
        let n = history_ids.len();
        if n >= 2 {
            self.p_kn_trigram(history_ids[n - 2], history_ids[n - 1], w_id)
        } else if n == 1 {
            self.p_kn_bigram(history_ids[0], w_id)
        } else {
            self.p_continuation(w_id)
        }
    }

    /// Natural log of the base (no-copy) probability of `token` given the history ids.
    ///
    /// Vocabulary words use the n-gram estimate. Other words use
    /// `P(<unk> | context) * P(spelling)` when `spelling` is `Some`, and the constant
    /// `1 / (V * 100)` otherwise.
    pub fn log_base_prob_text(&self, token: &str, history_ids: &[u32], spelling: Option<&SpellCache>) -> f64 {
        if let Some(id) = self.get_token_id(token) {
            return self.predict_base_prob_by_id(id, history_ids).ln();
        }
        match spelling {
            Some(cache) => {
                self.predict_base_prob_by_id(UNK_ID, history_ids).ln() + self.log_spell_cached(token, cache)
            }
            None => self.predict_base_prob_by_id(UNK_ID, history_ids).ln(),
        }
    }

    fn log_spell_cached(&self, token: &str, cache: &SpellCache) -> f64 {
        if let Some(&v) = cache.borrow().get(token) {
            return v;
        }
        let model = self
            .spelling
            .as_ref()
            .expect("spelling model requested but not trained for this sector");
        let v = model.log_prob(token);
        cache.borrow_mut().insert(token.to_string(), v);
        v
    }

    /// Predicts token probability with history and optional copy mechanism.
    /// Non-vocabulary history tokens are preserved as <unk> (ID 0).
    pub fn predict_token_prob(
        &self,
        token: &str,
        history: &[&str],
        symbol_table: Option<&SymbolTable>,
        copy_weight: f64,
    ) -> f64 {
        self.predict_token_prob_opts(token, history, symbol_table, copy_weight, None)
    }

    /// Natural log of the probability of `token`, including the copy mechanism.
    /// Computed in log space so that very long out-of-vocabulary words do not underflow.
    pub fn log_prob_token_opts(
        &self,
        token: &str,
        history: &[&str],
        symbol_table: Option<&SymbolTable>,
        copy_weight: f64,
        spelling: Option<&SpellCache>,
    ) -> f64 {
        let n_active = symbol_table.map_or(0, |s| s.active_ids().len());
        let history_ids: Vec<u32> = history.iter().map(|&h| self.get_token_id_or_unk(h)).collect();
        let lb = self.log_base_prob_text(token, &history_ids, spelling);
        if n_active > 0 && copy_weight > 0.0 {
            let a = (1.0 - copy_weight).ln() + lb;
            if symbol_table.is_some_and(|s| s.is_active_name(token)) {
                let b = (copy_weight / n_active as f64).ln();
                let m = a.max(b);
                m + ((a - m).exp() + (b - m).exp()).ln()
            } else {
                a
            }
        } else {
            lb
        }
    }

    /// Same as `predict_token_prob`, with an optional spelling model for unknown words.
    pub fn predict_token_prob_opts(
        &self,
        token: &str,
        history: &[&str],
        symbol_table: Option<&SymbolTable>,
        copy_weight: f64,
        spelling: Option<&SpellCache>,
    ) -> f64 {
        let n_active = symbol_table.map_or(0, |s| s.active_ids().len());
        let is_sym = symbol_table.is_some_and(|s| s.is_active_name(token));
        let history_ids: Vec<u32> = history.iter().map(|&h| self.get_token_id_or_unk(h)).collect();
        let base_prob = self.log_base_prob_text(token, &history_ids, spelling).exp();

        if n_active > 0 && copy_weight > 0.0 {
            let copy_prob = if is_sym { 1.0 / n_active as f64 } else { 0.0 };
            (1.0 - copy_weight) * base_prob + copy_weight * copy_prob
        } else {
            base_prob
        }
    }
    /// Exact top-k for both variants (with copy from the symbol table, and without)
    /// in one pass, with early stopping.
    ///
    /// Every token not yet visited has probability at most
    /// `λ3·((c_next − D)/c(u2) + λ2·P_cont_max)` without copy, and `(1−cw)` times that with
    /// copy (symbols are always scored explicitly, so they never hide among unvisited tokens).
    /// The scan stops only when the bound is below the k-th best score of BOTH variants.
    /// Each variant keeps a min-heap of size k, so reading the k-th score is O(1) and
    /// inserting a candidate is O(log k).
    ///
    /// Returns `(with_copy, without_copy)`. When there are no active symbols or
    /// `copy_weight <= 0`, both are identical.
    #[allow(clippy::type_complexity)]
    pub fn top_k_dual(
        &self,
        history: &[&str],
        k: usize,
        symbols: Option<&SymbolTable>,
        copy_weight: f64,
    ) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
        self.top_k_dual_opts(history, k, symbols, copy_weight, None)
    }

    /// 	op_k_dual with an optional spelling model for out-of-vocabulary active symbols.
    #[allow(clippy::type_complexity)]
    pub fn top_k_dual_opts(
        &self,
        history: &[&str],
        k: usize,
        symbols: Option<&SymbolTable>,
        copy_weight: f64,
        spelling: Option<&SpellCache>,
    ) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
        if self.vocab.is_empty() || k == 0 {
            return (Vec::new(), Vec::new());
        }

        let history_ids: Vec<u32> = history.iter().map(|&h| self.get_token_id_or_unk(h)).collect();
        let n = history_ids.len();

        let n_active = symbols.map_or(0, |s| s.active_ids().len());
        // Same rule as predict_token_prob: copy only applies with active symbols.
        let cw = if n_active > 0 && copy_weight > 0.0 { copy_weight } else { 0.0 };
        let copy_bonus = if cw > 0.0 { cw / n_active as f64 } else { 0.0 };

        let mut heap_copy: BinaryHeap<Reverse<Cand>> = BinaryHeap::with_capacity(k + 1);
        let mut heap_base: BinaryHeap<Reverse<Cand>> = BinaryHeap::with_capacity(k + 1);
        let mut seen_ids: HashSet<u32> = HashSet::new();
        // Vocabulary ids of active symbols (marks which candidates get the copy bonus)
        let mut sym_vocab_ids: HashSet<u32> = HashSet::new();

        fn push_top_k<'a>(heap: &mut BinaryHeap<Reverse<Cand<'a>>>, k: usize, c: Cand<'a>) {
            if heap.len() < k {
                heap.push(Reverse(c));
            } else if let Some(worst) = heap.peek() {
                if c > worst.0 {
                    heap.pop();
                    heap.push(Reverse(c));
                }
            }
        }

        // Active symbols: in-vocabulary ones are scored through the model below,
        // out-of-vocabulary ones get the uniform fallback probability.
        let mut oov_symbols: Vec<&str> = Vec::new();
        if let Some(st) = symbols {
            for &sid in st.active_ids() {
                let name = st.name(sid);
                match self.get_token_id(name) {
                    Some(vid) => {
                        sym_vocab_ids.insert(vid);
                    }
                    None => oov_symbols.push(name),
                }
            }
        }

        if !oov_symbols.is_empty() {
            for name in &oov_symbols {
                let base_p = self.log_base_prob_text(name, &history_ids, spelling).exp();
                let copy_p = (1.0 - cw) * base_p + copy_bonus;
                push_top_k(&mut heap_copy, k, Cand { score: copy_p, text: name, id: u32::MAX });
                push_top_k(&mut heap_base, k, Cand { score: base_p, text: name, id: u32::MAX });
            }
        }

        macro_rules! record {
            ($cid:expr) => {{
                let cid: u32 = $cid;
                if seen_ids.insert(cid) {
                    let base_prob = self.predict_base_prob_by_id(cid, &history_ids);
                    let with_copy = if sym_vocab_ids.contains(&cid) {
                        (1.0 - cw) * base_prob + copy_bonus
                    } else {
                        (1.0 - cw) * base_prob
                    };
                    let text = self.vocab[cid as usize].as_str();
                    push_top_k(&mut heap_copy, k, Cand { score: with_copy, text, id: cid });
                    push_top_k(&mut heap_base, k, Cand { score: base_prob, text, id: cid });
                }
            }};
        }
        // 1. Trigram continuations
        if n >= 2 {
            let u1 = history_ids[n - 2];
            let u2 = history_ids[n - 1];
            if let Some(conts) = self.trigram_conts_sorted.get(&(u1, u2)) {
                for &(w, _) in conts {
                    record!(w);
                }
            }
        }

        // 2. Active symbols present in the vocabulary
        for &vid in &sym_vocab_ids {
            record!(vid);
        }

        // Trigram backoff weight lambda_3 and the lower-order context u2
        let (lambda_3, u2) = if n >= 2 {
            let u1 = history_ids[n - 2];
            let u2 = history_ids[n - 1];
            let c_u12 = self.ctx2_total.get(&(u1, u2)).copied().unwrap_or(0) as f64;
            if c_u12 > 0.0 {
                let num_f = self.distinct_followers_2.get(&(u1, u2)).copied().unwrap_or(0) as f64;
                ((self.discount / c_u12) * num_f, u2)
            } else {
                (1.0, u2)
            }
        } else if n == 1 {
            (1.0, history_ids[0])
        } else {
            (1.0, UNK_ID)
        };

        // The scan may stop when no unvisited token can enter either top-k.
        let can_stop = |bound: f64,
                        heap_copy: &BinaryHeap<Reverse<Cand>>,
                        heap_base: &BinaryHeap<Reverse<Cand>>| {
            heap_copy.len() >= k
                && heap_base.len() >= k
                && (1.0 - cw) * bound < heap_copy.peek().map_or(f64::NEG_INFINITY, |c| c.0.score)
                && bound < heap_base.peek().map_or(f64::NEG_INFINITY, |c| c.0.score)
        };

        // 3. Bigram continuations (descending count) with early stopping
        let c_u2 = self.ctx1_total.get(&u2).copied().unwrap_or(0) as f64;
        let num_f2 = self.distinct_followers_1.get(&u2).copied().unwrap_or(0) as f64;
        let lambda_2 = if c_u2 > 0.0 { (self.discount / c_u2) * num_f2 } else { 1.0 };

        if n >= 1 {
            if let Some(conts) = self.bigram_conts_sorted.get(&u2) {
                for &(w, c_next) in conts {
                    let p_b2_bound = if c_u2 > 0.0 {
                        (c_next as f64 - self.discount).max(0.0) / c_u2 + lambda_2 * self.max_p_cont
                    } else {
                        self.max_p_cont
                    };
                    if can_stop(lambda_3 * p_b2_bound, &heap_copy, &heap_base) {
                        break;
                    }
                    record!(w);
                }
            }
        }

        // 4. Unigram continuation backoff with early stopping
        for &w in &self.unigram_sorted_by_cont {
            if can_stop(lambda_3 * lambda_2 * self.p_continuation(w), &heap_copy, &heap_base) {
                break;
            }
            record!(w);
        }

        let drain = |heap: BinaryHeap<Reverse<Cand>>| -> Vec<(String, f64)> {
            // into_sorted_vec on Reverse<Cand> is ascending in Reverse, i.e. best first
            heap.into_sorted_vec()
                .into_iter()
                .map(|Reverse(c)| (c.text.to_string(), c.score))
                .collect()
        };

        (drain(heap_copy), drain(heap_base))
    }

    /// Convenience top_k
    pub fn top_k(
        &self,
        history: &[&str],
        k: usize,
        symbol_table: Option<&SymbolTable>,
        copy_weight: f64,
    ) -> Vec<(String, f64)> {
        let (with_copy, no_copy) = self.top_k_dual(history, k, symbol_table, copy_weight);
        let has_symbols = symbol_table.is_some_and(|s| !s.active_ids().is_empty());
        if copy_weight > 0.0 && has_symbols {
            with_copy
        } else {
            no_copy
        }
    }

    /// Reference implementation: exhaustive scan of full vocabulary.
    pub fn top_k_exhaustive(
        &self,
        history: &[&str],
        k: usize,
        symbol_table: Option<&SymbolTable>,
        copy_weight: f64,
    ) -> Vec<(String, f64)> {
        let mut candidates: Vec<(String, f64)> = Vec::with_capacity(self.vocab.len());
        let mut seen = HashSet::new();

        for text in &self.vocab {
            let prob = self.predict_token_prob(text, history, symbol_table, copy_weight);
            candidates.push((text.clone(), prob));
            seen.insert(text.clone());
        }

        if let Some(syms) = symbol_table {
            for name in syms.active_symbol_names() {
                if !seen.contains(&name) {
                    let prob = self.predict_token_prob(&name, history, symbol_table, copy_weight);
                    candidates.push((name, prob));
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(k);
        candidates
    }

    pub fn to_ngram_counts(&self) -> SectorNgramCounts {
        SectorNgramCounts {
            unigrams: self.unigram_counts.clone(),
            bigrams: self.bigram_counts.clone(),
            trigrams: self.trigram_counts.clone(),
            total_unigrams: self.total_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::SectorKind;

    #[test]
    fn test_kneser_ney_smoothing_and_copy_mechanism() {
        let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());
        model.train(&["x", "y", "x", "z", "x", "y"]);
        model.recompute_continuation_counts();

        assert!(model.p_kn_bigram(model.get_token_id("x").unwrap(), model.get_token_id("y").unwrap()) > 0.0);

        let mut sym_table = SymbolTable::new();
        sym_table.define("Animal", SectorKind::DefinedName.id(), 'G', 0);

        let prob_no_copy = model.predict_token_prob("Animal", &["x"], None, 0.0);
        let prob_with_copy = model.predict_token_prob("Animal", &["x"], Some(&sym_table), 0.5);

        assert!(prob_with_copy > prob_no_copy * 10.0);

        let top5 = model.top_k(&["x"], 5, Some(&sym_table), 0.5);
        let top5_words: Vec<&str> = top5.iter().map(|(s, _)| s.as_str()).collect();
        assert!(top5_words.contains(&"Animal"));
    }

    #[test]
    fn test_unknown_token_in_context_not_skipped() {
        let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());
        model.train(&["foo", "bar", "baz"]);
        model.train(&["foo", "bar", "baz"]);
        model.recompute_continuation_counts();

        let prob_adjacent = model.predict_token_prob("baz", &["foo", "bar"], None, 0.0);
        let prob_with_unk_middle = model.predict_token_prob(
            "baz",
            &["foo", "UNKNOWN_NONEXISTENT_MIDDLE", "bar"],
            None,
            0.0,
        );

        assert!(
            prob_adjacent > prob_with_unk_middle,
            "Context with unknown token in the middle ({}) must NOT predict like adjacent context ({})",
            prob_with_unk_middle,
            prob_adjacent
        );
    }

    #[test]
    fn singleton_types_train_unk_mass_at_their_corpus_rate() {
        let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());
        let mut singletons = HashSet::new();
        let mut stream = Vec::new();
        for i in 0..20 {
            stream.push("common".to_string());
            let rare = format!("rare_{i}");
            singletons.insert(rare.clone());
            stream.push(rare);
        }
        model.set_singleton_tokens(singletons);
        let refs: Vec<&str> = stream.iter().map(String::as_str).collect();
        model.train(&refs);
        model.recompute_continuation_counts();
        let p_unk = model.p_continuation(UNK_ID);
        let singleton_fraction = 0.5;
        assert!((0.25..=0.75).contains(&p_unk), "P(<unk>)={p_unk}, singleton fraction={singleton_fraction}");
        assert_eq!(model.get_token_id("rare_0"), None, "singletons should route through <unk> in n-grams while remaining stored for spelling");
        assert!(model.vocab_indices.contains_key("rare_0"));
        for history in [&[][..], &["common"][..], &["absent"][..]] {
            let ids: Vec<u32> = history.iter().map(|s| model.get_token_id_or_unk(s)).collect();
            let total: f64 = model.ngram_vocab_ids.iter().map(|&id| model.predict_base_prob_by_id(id, &ids)).sum();
            assert!((total - 1.0).abs() < 1e-9, "singleton-aware probabilities sum to {total} in {history:?}");
        }
    }

    #[test]
    fn mixed_event_maps_singleton_same_sector_context_to_unk() {
        let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());
        model.set_singleton_tokens(["rare".to_string()].into_iter().collect());
        model.train_event(&[], "rare");
        model.train_event(&["rare", "<sec:08>"], "target");
        let unk = UNK_ID;
        let marker = model.get_token_id("<sec:08>").unwrap();
        let target = model.get_token_id("target").unwrap();
        assert!(model.trigram_counts.contains_key(&(unk, marker, target)));
        let rare_id = model.vocab_indices["rare"];
        assert!(!model.trigram_counts.contains_key(&(rare_id, marker, target)));
    }

    #[test]
    fn test_sector_probabilities_sum_to_one_across_all_sectors_and_contexts() {
        use crate::sector::parser::PythonAstParser;
        use crate::sector::NUM_SECTORS;

        let mut parser = PythonAstParser::new();
        let samples = [
            "import os\nimport sys\n\ndef compute(a, b=10):\n    # comment\n    x = a + b * 2\n    return str(x)\n",
            "class Animal:\n    @property\n    def name(self):\n        return 'dog'\n",
            "for i in range(5):\n    if i > 2:\n        print(i)\n",
        ];

        let mut models: Vec<SectorKneserNeyModel> = (0..NUM_SECTORS as u8)
            .map(SectorKneserNeyModel::new)
            .collect();

        for code in &samples {
            let toks = parser.parse_code_with_format(code).expect("valid python");
            let mut streams: Vec<Vec<&str>> = vec![Vec::new(); NUM_SECTORS];
            for t in &toks {
                streams[t.sector.id() as usize].push(t.text.as_str());
            }
            for (sid, s) in streams.into_iter().enumerate() {
                models[sid].train(&s);
            }
        }
        for m in &mut models {
            m.recompute_continuation_counts();
        }

        let test_contexts: &[&[&str]] = &[
            &[],
            &["def"],
            &["self"],
            &["."],
            &["\n    "],
            &["def", "compute"],
            &["a", "+"],
            &["UNSEEN_TOKEN"],
            &["def", "UNSEEN_TOKEN"],
            &["UNSEEN_1", "UNSEEN_2"],
        ];

        for (sid, model) in models.iter().enumerate() {
            for &ctx in test_contexts {
                let hist_ids: Vec<u32> = ctx.iter().map(|&h| model.get_token_id_or_unk(h)).collect();
                // Sum over known vocabulary (1..V) + <unk> mass (ID 0)
                let vocab_mass: f64 = model.ngram_vocab_ids.iter().copied().filter(|&id| id != UNK_ID)
                    .map(|id| model.predict_base_prob_by_id(id, &hist_ids))
                    .sum();
                let unk_mass = model.predict_base_prob_by_id(UNK_ID, &hist_ids);
                let total = vocab_mass + unk_mass;
                assert!(
                    (total - 1.0).abs() < 1e-9,
                    "Sector {} in context {:?}: vocab ({}) + <unk> ({}) = {} != 1.0",
                    sid,
                    ctx,
                    vocab_mass,
                    unk_mass,
                    total
                );
            }
        }

        // Verify that an unseen token gets P(<unk> | ctx) * P(spelling)
        let spell_cache = SpellCache::default();
        let m_used = &models[SectorKind::UsedName.id() as usize];
        let ctx = &["a"];
        let hist_ids: Vec<u32> = ctx.iter().map(|&h| m_used.get_token_id_or_unk(h)).collect();
        let unk_p = m_used.predict_base_prob_by_id(UNK_ID, &hist_ids);
        let spell_lp = m_used.spelling.as_ref().unwrap().log_prob("brand_new_var");
        let combined_lp = m_used.log_base_prob_text("brand_new_var", &hist_ids, Some(&spell_cache));
        assert!(
            (combined_lp - (unk_p.ln() + spell_lp)).abs() < 1e-12,
            "Unseen token log-prob must equal ln P(<unk>|ctx) + ln P(spelling)"
        );
    }
}
