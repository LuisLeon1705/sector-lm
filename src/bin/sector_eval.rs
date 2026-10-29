use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use clap::Parser as ClapParser;
use rayon::prelude::*;
use serde_json::json;

use sector_lm::data::{DatasetItem, load_dataset_jsonl, verify_no_leakage};
use sector_lm::eval::{
    evaluate_dataset_sector_lm_parallel_opts, text_category, EvalOptions,
    EvaluatorAccumulator, ModelMetrics, ProgressTracker,
};
use sector_lm::model::baseline_bpe::BpeNgramBaseline;
use sector_lm::model::hmm::SectorHmm;
use sector_lm::model::ngram::SectorKneserNeyModel;
use sector_lm::sector::parser::PythonAstParser;
use sector_lm::sector::parser::ParsedToken;
use sector_lm::sector::{SectorKind, NUM_SECTORS};
use sector_lm::sector::domain::{remap_tokens_to_domain, NUM_DOMAIN_SECTORS};
use sector_lm::model::text_bpe::{TextBpe, text_contents};
use sector_lm::model::history::{HistoryMode, training_events};
use sector_lm::storage::cache::SectorLruCache;
use sector_lm::storage::sector_file::{SectorFileData, SectorFileStorage};

#[derive(ClapParser, Debug)]
#[command(author, version, about = "Sector-LM Evaluation and Comparative Benchmarking")]
struct Args {
    #[arg(short, long, default_value = "data/train.jsonl")]
    train_file: String,

    #[arg(short, long, default_value = "data/test.jsonl")]
    test_file: String,

    #[arg(long, default_value = "results/starcoder_metrics.json")]
    starcoder_metrics: String,

    #[arg(long, default_value = "results/benchmark_summary.json")]
    output_summary: String,

    #[arg(long, default_value_t = 100)]
    max_test_files: usize,

    /// Evaluate only the test file at this index (0-based, in the order of the test file).
    /// Skips the StarCoder comparison, whose metrics cover the full set.
    #[arg(long)]
    only_file: Option<usize>,

    #[arg(long, default_value = "sectors_storage")]
    storage_dir: String,

    #[arg(long, default_value_t = 4)]
    cache_capacity: usize,

    #[arg(long, default_value_t = 0.4)]
    copy_weight: f64,

    #[arg(long, default_value_t = 16000)]
    bpe_vocab_size: usize,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    price_sector: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    format: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    spelling: bool,
    #[arg(long, default_value = "domain", value_parser = ["syntactic", "domain"])]
    sectors: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    unk_singletons: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    text_bpe: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    spelling_frequency: bool,
    #[arg(long, default_value_t = 8000)]
    text_bpe_vocab_size: usize,
    #[arg(long, value_enum, default_value_t = HistoryMode::Sector)]
    history: HistoryMode,
}

fn create_progress_bar(total: u64, desc: &str, unit: &'static str) -> ProgressTracker {
    ProgressTracker::new(desc, unit, total)
}

fn category_metrics(acc: &EvaluatorAccumulator) -> serde_json::Value {
    let names = ["code", "comments", "strings", "format"];
    let mut out = serde_json::Map::new();
    for i in 0..4 {
        let chars = acc.category_chars[i];
        let nats_per_char = if chars == 0 { 0.0 } else { acc.category_nats[i] / chars as f64 };
        out.insert(names[i].to_string(), json!({"chars": chars, "nats_per_char": nats_per_char, "perplexity_per_char": nats_per_char.exp()}));
    }
    serde_json::Value::Object(out)
}

fn main() {
    let total_eval_start = Instant::now();
    let args = Args::parse();

    println!("================================================================================");
    println!("                   SECTOR-LM: PROTOTYPE EVALUATION & BENCHMARK                  ");
    println!("================================================================================");

    // 1. Load Datasets
    println!("\n[1/6] Loading datasets...");
    let train_items = load_dataset_jsonl(&args.train_file)
        .unwrap_or_else(|e| panic!("Failed to load train file {}: {}", args.train_file, e));
    let test_items_all = load_dataset_jsonl(&args.test_file)
        .unwrap_or_else(|e| panic!("Failed to load test file {}: {}", args.test_file, e));

    let test_items: Vec<DatasetItem> = match args.only_file {
        Some(i) => {
            let item = test_items_all
                .get(i)
                .unwrap_or_else(|| panic!("--only-file {} out of range ({} test files)", i, test_items_all.len()))
                .clone();
            vec![item]
        }
        None => test_items_all.into_iter().take(args.max_test_files).collect(),
    };

    let total_test_chars: usize = test_items.iter().map(|it| it.content.chars().count()).sum();
    println!("Train files: {}", train_items.len());
    println!("Test files:  {} (Exact Total Characters: {})", test_items.len(), total_test_chars);

    // 2. Data Leakage Verification
    println!("\n[2/6] Verifying zero data leakage between train and test...");
    let (is_clean, leaked) = verify_no_leakage(&train_items, &test_items);
    if !is_clean {
        panic!("FATAL: Found {} leaked files in train set! Hashes: {:?}", leaked.len(), leaked);
    }
    println!("VERIFIED: 0 test files appear in training set (SHA-256 collision check).");

    // 3. Training Sector-LM (HMM + Sector N-Grams)
    println!("\n[3/6] Training Sector-LM on {} files...", train_items.len());
    let train_start = Instant::now();

    let pb_parse = create_progress_bar(train_items.len() as u64, "Parse AST", "files");
    let raw_train: Vec<Vec<ParsedToken>> = train_items
        .par_iter()
        .filter_map(|item| {
            let mut parser = PythonAstParser::new();
            let tokens = if args.format { parser.parse_code_with_format(&item.content)? } else { parser.parse_code(&item.content)? };
            pb_parse.inc(1);
            Some(tokens)
        })
        .collect();
    pb_parse.finish();

    println!("Successfully parsed {} files with tree-sitter AST.", raw_train.len());
    let mut text_training_spans = 0usize;
    let text_tokenizer = if args.text_bpe {
        let corpus: Vec<String> = raw_train.iter().flat_map(|seq| text_contents(seq)).collect();
        text_training_spans = corpus.len();
        println!("Training comment/string-content BPE on {} content spans...", corpus.len());
        Some(TextBpe::train(&corpus, args.text_bpe_vocab_size).unwrap_or_else(|e| panic!("Text BPE training failed: {e}")))
    } else { None };
    let parsed_train: Vec<Vec<ParsedToken>> = raw_train.into_iter().map(|seq| {
        let split = if let Some(bpe) = &text_tokenizer { bpe.tokenize(&seq).expect("split training text tokens") } else { seq };
        if args.sectors == "domain" { remap_tokens_to_domain(&split).1 } else { split }
    }).collect();

    // Fit HMM
    println!("Fitting Sector Hidden Markov Model (HMM)...");
    let mut hmm = SectorHmm::new();
    let hmm_sequences: Vec<Vec<(u8, String)>> = parsed_train
        .iter()
        .map(|seq| seq.iter().map(|t| (t.sector.id(), t.text.clone())).collect())
        .collect();
    hmm.fit(&hmm_sequences);

    // Train Sector N-gram models
    println!("Training sector-specific Kneser-Ney N-gram models ({} sectors)...", if args.sectors == "domain" { NUM_DOMAIN_SECTORS } else { NUM_SECTORS });
    let active_sector_count = NUM_SECTORS;
    let mut sector_models: Vec<SectorKneserNeyModel> = (0..active_sector_count as u8)
        .map(SectorKneserNeyModel::new)
        .collect();

    if args.unk_singletons {
        let mut frequencies: Vec<std::collections::HashMap<String, u64>> =
            (0..active_sector_count).map(|_| std::collections::HashMap::new()).collect();
        for seq in &parsed_train {
            for token in seq {
                *frequencies[token.sector.id() as usize].entry(token.text.clone()).or_default() += 1;
            }
        }
        for (model, counts) in sector_models.iter_mut().zip(frequencies) {
            model.set_singleton_tokens(counts.into_iter().filter_map(|(word, count)| (count == 1).then_some(word)).collect());
        }
    }

    let pb_train_sec = create_progress_bar(parsed_train.len() as u64, "Train Sectors", "files");
    for seq in &parsed_train {
        let mut sector_streams: Vec<Vec<&str>> = vec![Vec::new(); NUM_SECTORS];
        if args.history == HistoryMode::Sector {
            for token in seq {
                let sid = token.sector.id() as usize;
                if sid < NUM_SECTORS { sector_streams[sid].push(token.text.as_str()); }
            }
            for (sid, stream) in sector_streams.into_iter().enumerate() {
                if !stream.is_empty() { sector_models[sid].train(&stream); }
            }
        } else {
            for event in training_events(seq, HistoryMode::Mixed) {
                let history: Vec<&str> = event.history.iter().map(String::as_str).collect();
                sector_models[event.target_sector as usize].train_event(&history, &event.target);
            }
        }
        pb_train_sec.inc(1);
    }
    pb_train_sec.finish();

    println!("Recomputing Kneser-Ney continuation tables...");
    for model in &mut sector_models {
        model.set_frequency_weighted_spelling(args.spelling_frequency);
        model.recompute_continuation_counts();
    }

    let train_elapsed = train_start.elapsed();
    println!("Sector-LM trained in {:.2}s.", train_elapsed.as_secs_f64());

    // Save sectors to disk as binary files
    println!("Serializing sectors to disk with binary format...");
    let storage_dir = Path::new(&args.storage_dir);
    let storage = Arc::new(SectorFileStorage::new(storage_dir).unwrap());

    let mut total_storage_bytes = 0;
    for (sid, model) in sector_models.iter().enumerate() {
        let sec_kind = SectorKind::from_id(sid as u8).unwrap();
        let sec_data = SectorFileData {
            sector_id: sid as u8,
            sector_name: sec_kind.name().to_string(),
            vocab: model.vocab.clone(),
            ngram_counts: model.to_ngram_counts(),
            total_tokens_observed: model.total_tokens,
        };
        let bytes = storage.save_sector(&sec_data).unwrap();
        total_storage_bytes += bytes;
    }
    println!(
        "Saved {} binary sector files to '{}' (total {:.2} MB on disk).",
        sector_models.len(),
        args.storage_dir,
        total_storage_bytes as f64 / (1024.0 * 1024.0)
    );

    // 4. Train Baseline (a): Monolithic N-Gram with standard BPE
    println!("\n[4/6] Training Baseline (a): Monolithic N-Gram with standard BPE tokenizer...");
    let bpe_start = Instant::now();
    let train_texts: Vec<&str> = train_items.iter().map(|it| it.content.as_str()).collect();
    println!("Training BPE tokenizer (vocab {}) and the monolithic n-gram model...", args.bpe_vocab_size);
    let bpe = BpeNgramBaseline::train(&train_texts, args.bpe_vocab_size)
        .unwrap_or_else(|e| panic!("BPE baseline training failed: {e}"));
    println!(
        "BPE tokenizer vocabulary: {} tokens; n-gram vocabulary seen in training: {}",
        bpe.tokenizer.get_vocab_size(false),
        bpe.model.vocab.len()
    );
    assert!(
        bpe.tokenizer.get_vocab_size(false) > 256,
        "BPE tokenizer has no learned merges"
    );    println!("Monolithic BPE N-Gram trained in {:.2}s.", bpe_start.elapsed().as_secs_f64());

    // 5. Evaluate on Test Set
    println!("\n[5/6] Evaluating on {} test files in parallel (Rayon)...", test_items.len());

    let mut test_parsed = Vec::new();
    let mut parser = PythonAstParser::new();
    for it in &test_items {
        if let Some(toks) = if args.format { parser.parse_code_with_format(&it.content) } else { parser.parse_code(&it.content) } {
            let toks = if let Some(bpe) = &text_tokenizer { bpe.tokenize(&toks).expect("split test text tokens") } else { toks };
            let toks = if args.sectors == "domain" { remap_tokens_to_domain(&toks).1 } else { toks };
            test_parsed.push((it.content.clone(), toks));
        }
    }

    let total_eval_tokens: usize = test_parsed.iter().map(|(_, toks)| toks.len().saturating_sub(1)).sum();
    let evaluated_chars: usize = test_parsed.iter().map(|(content, _)| content.chars().count()).sum();
    assert_eq!(evaluated_chars, total_test_chars, "parsed test files do not cover the full test character set");
    if args.only_file.is_none() { assert_eq!(evaluated_chars, 795_700, "expected exactly 795,700 test characters"); }
    println!("Total tokens to evaluate in Sector-LM: {}", total_eval_tokens);

    let sector_eval_start = Instant::now();
    let mut sector_cache = SectorLruCache::new(Arc::clone(&storage), args.cache_capacity);

    let pb_sector_eval = create_progress_bar(total_eval_tokens as u64, "Sector-LM Eval", "tokens");

    let (mut sector_eval_accum, sector_eval_without_copy, viterbi_correct, viterbi_total, file_results) =
        evaluate_dataset_sector_lm_parallel_opts(
            &test_parsed,
            &hmm,
            &sector_models,
            args.copy_weight,
            EvalOptions { price_sector: args.price_sector, use_spelling: args.spelling, fixed_sectors: args.sectors == "domain", history_mode: args.history },
            &mut sector_cache,
            Some(&pb_sector_eval),
        );
    pb_sector_eval.finish();

    let sector_eval_elapsed = sector_eval_start.elapsed();
    sector_eval_accum.elapsed_secs = sector_eval_elapsed.as_secs_f64();

    let viterbi_acc = if viterbi_total > 0 {
        viterbi_correct as f64 / viterbi_total as f64
    } else {
        0.0
    };

    let cache_stats = sector_cache.stats();
    // Real resident memory in MB directly from resident sector bytes!
    let sector_lm_mem_mb = sector_cache.resident_memory_mb();

    let sector_lm_metrics = sector_eval_accum.compute_metrics(
        if args.sectors == "domain" { "Sector-LM (domain + text BPE + symboles)" } else { "Sector-LM (syntactic + text BPE + symboles)" },
        sector_lm_mem_mb,
        Some(cache_stats.sectors_loaded_per_1000_tokens()),
        Some(cache_stats.bytes_loaded_per_1000_tokens()),
        Some(viterbi_acc),
    );
    println!("     Sector-LM evaluation completed in {:.2}s.", sector_eval_elapsed.as_secs_f64());
    println!("   PPC by text type (nats/char): code {:.4}, comments {:.4}, strings {:.4}, format {:.4}",
        sector_eval_accum.category_nats[0] / sector_eval_accum.category_chars[0].max(1) as f64,
        sector_eval_accum.category_nats[1] / sector_eval_accum.category_chars[1].max(1) as f64,
        sector_eval_accum.category_nats[2] / sector_eval_accum.category_chars[2].max(1) as f64,
        sector_eval_accum.category_nats[3] / sector_eval_accum.category_chars[3].max(1) as f64);

    // Analyze slowest files
    let mut sorted_by_time = file_results.clone();
    sorted_by_time.sort_by(|a, b| b.elapsed_secs.partial_cmp(&a.elapsed_secs).unwrap());
    println!("\nTop 5 slowest files evaluated by Sector-LM:");
    for (rank, f) in sorted_by_time.iter().take(5).enumerate() {
        println!(
            "  #{}: File idx {}, {} tokens, {} chars -> {:.2}s",
            rank + 1, f.file_index, f.token_count, f.exact_char_count, f.elapsed_secs
        );
        if f.elapsed_secs > 60.0 {
            println!("  [WARNING] File idx {} exceeded 60 seconds threshold!", f.file_index);
        }
    }

    // Evaluate Baseline (a): Monolithic BPE in parallel
    println!("\n  -> Evaluating Baseline (a): Monolithic BPE in parallel...");
    let bpe_eval_start = Instant::now();

    let bpe_encoded: Vec<Vec<(String, usize, usize)>> = test_parsed
        .iter()
        .map(|(c, _)| bpe.encode_pieces_with_offsets(c).expect("BPE encode of test file"))
        .collect();
    let total_bpe_eval_tokens: usize = bpe_encoded.iter().map(|p| p.len().saturating_sub(1)).sum();
    assert!(total_bpe_eval_tokens > 0, "BPE baseline produced no tokens to evaluate");
    println!("Total tokens to evaluate in BPE baseline: {}", total_bpe_eval_tokens);
    let pb_bpe_eval = create_progress_bar(total_bpe_eval_tokens as u64, "BPE Eval", "tokens");

    let bpe_accum_results: Vec<EvaluatorAccumulator> = test_parsed
        .par_iter()
        .zip(bpe_encoded.par_iter())
        .map(|((content, tokens), pieces)| {
            let mut accum = EvaluatorAccumulator::new();
            accum.total_chars = content.chars().count() as u64;
            let mut char_categories = vec![0usize; accum.total_chars as usize];
            for tok in tokens {
                let start = content.get(..tok.byte_range.0).map_or(0, |s| s.chars().count());
                let category = text_category(tok);
                for c in char_categories.iter_mut().skip(start).take(tok.text.chars().count()) { *c = category; }
            }
            for &category in &char_categories { accum.category_chars[category] += 1; }
            let mut batch = 0u64;
            for i in 1..pieces.len() {
                let start = i.saturating_sub(2);
                let history: Vec<&str> = pieces[start..i].iter().map(|s| s.0.as_str()).collect();
                let (target, off_start, off_end) = (&pieces[i].0, pieces[i].1, pieces[i].2);
                let target_prob = bpe.model.predict_token_prob(target, &history, None, 0.0);
                let top5 = bpe.model.top_k(&history, 5, None, 0.0);
                accum.record_prediction(target, &top5, target_prob, false);
                let mut counts = [0usize; 4];
                for &c in char_categories.iter().take(off_end).skip(off_start) { counts[c] += 1; }
                let category = counts.iter().enumerate().max_by_key(|(_, n)| *n).map_or(0, |(i, _)| i);
                accum.category_nats[category] += -target_prob.max(1e-12).ln();
                batch += 1;
                if batch >= 256 {
                    pb_bpe_eval.inc(batch);
                    batch = 0;
                }
            }
            pb_bpe_eval.inc(batch);
            accum
        })
        .collect();
    pb_bpe_eval.finish();

    let mut bpe_accum = EvaluatorAccumulator::new();
    for acc in bpe_accum_results {
        bpe_accum.merge(&acc);
    }
    let bpe_eval_elapsed = bpe_eval_start.elapsed();
    bpe_accum.elapsed_secs = bpe_eval_elapsed.as_secs_f64();

    // Serialized size of vocabulary + n-gram counts: the same content a sector file holds.
    let bpe_mem_mb = bincode::serialized_size(&(&bpe.model.vocab, bpe.model.to_ngram_counts()))
        .expect("serialize BPE model") as f64
        / (1024.0 * 1024.0);
    let bpe_metrics = bpe_accum.compute_metrics(
        "Línea base (a): N-gramas monolíticos BPE",
        bpe_mem_mb,
        None,
        None,
        None,
    );
    let sector_type_metrics = category_metrics(&sector_eval_accum);
    let bpe_type_metrics = category_metrics(&bpe_accum);
    let category_chars_total: u64 = sector_eval_accum.category_chars.iter().sum();
    let bpe_category_chars_total: u64 = bpe_accum.category_chars.iter().sum();
    assert_eq!(category_chars_total, total_test_chars as u64, "Sector-LM type breakdown does not cover all test characters");
    assert_eq!(bpe_category_chars_total, total_test_chars as u64, "BPE type breakdown does not cover all test characters");
    println!("Sector-LM unseen spelling model: {:.3} nats/char over {} unseen-token characters", sector_eval_accum.unseen_spelling_nats / sector_eval_accum.unseen_spelling_chars.max(1) as f64, sector_eval_accum.unseen_spelling_chars);
    println!("     Monolithic BPE evaluation completed in {:.2}s.", bpe_eval_elapsed.as_secs_f64());

    // Load Baseline (b): tiny_starcoder_py
    println!("  -> Loading Baseline (b): tiny_starcoder_py metrics from '{}'...", args.starcoder_metrics);
    let starcoder_metrics: Option<ModelMetrics> = if args.only_file.is_some() {
        None
    } else if Path::new(&args.starcoder_metrics).exists() {
        let f = File::open(&args.starcoder_metrics).expect("Open starcoder metrics file");
        let reader = BufReader::new(f);
        // Schema written by scripts/eval_starcoder.py
        let v: serde_json::Value = serde_json::from_reader(reader).expect("Parse starcoder metrics");
        let num = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("starcoder metrics: missing '{}'", k));
        Some(ModelMetrics {
            model_name: v["model_name"].as_str().expect("model_name").to_string(),
            top1_accuracy: num("top1_accuracy"),
            top5_accuracy: num("top5_accuracy"),
            perplexity_per_char: num("perplexity_per_char"),
            memory_mb: num("memory_mb"),
            tokens_per_sec: num("tokens_per_sec"),
            sectors_loaded_per_1000_tokens: None,
            bytes_loaded_per_1000_tokens: None,
            symbol_resolution_accuracy: None,
            sector_disambiguation_accuracy: None,
            exact_total_chars: v["total_chars_evaluated"].as_u64().expect("total_chars_evaluated"),
        })
    } else {
        panic!("Missing starcoder metrics file at '{}'!", args.starcoder_metrics);
    };

    // 6. Comparative Table
    let total_eval_time_secs = total_eval_start.elapsed().as_secs_f64();
    println!("\n[6/6] COMPARATIVE EVALUATION RESULTS (LÍNEAS BASE)");
    println!("=========================================================================================================");
    println!("TEXTO EVALUADO: EXACTAMENTE LOS MISMOS 100 ARCHIVOS DE TEST (TOTAL EXACTO: {} CARACTERES)", total_test_chars);
    println!("TIEMPO TOTAL DEL BENCHMARK: {:.2}s", total_eval_time_secs);
    println!("NOTA: PPC (Perplejidad por Carácter) es la métrica principal directamente comparable entre tokenizadores.");
    println!("      Top-1 / Top-5 están medidas por token de cada modelo y NO son directamente comparables entre tokenizadores.");
    println!("=========================================================================================================");
    println!(
        "{:<38} | {:<10} | {:<10} | {:<10} | {:<12} | {:<12}",
        "Modelo", "Top-1 Acc*", "Top-5 Acc*", "PPC (Char)", "Memoria (MB)", "Velocidad"
    );
    println!("---------------------------------------------------------------------------------------------------------");

    let print_row = |m: &ModelMetrics| {
        println!(
            "{:<38} | {:>9.2}% | {:>9.2}% | {:>10.3} | {:>10.2} MB | {:>8.0} tok/s",
            m.model_name,
            m.top1_accuracy * 100.0,
            m.top5_accuracy * 100.0,
            m.perplexity_per_char,
            m.memory_mb,
            m.tokens_per_sec
        );
    };

    print_row(&sector_lm_metrics);
    print_row(&bpe_metrics);
    if let Some(sc) = &starcoder_metrics {
        print_row(sc);
    }
    println!("=========================================================================================================");

    println!("\nDETAILED MEASUREMENTS (SECTOR-LM SPECIFICS):");
    println!("---------------------------------------------------------------------------------------------------------");
    println!("1. HMM Viterbi Sector Disambiguation:");
    println!("   - Precisión recuperando el sector correcto vs Tree-sitter: {:.2}% ({}/{} tokens)",
             viterbi_acc * 100.0, viterbi_correct, viterbi_total);
    println!("2. Vinculación y Tabla de Símbolos (Sector 07):");
    let acc_with = sector_lm_metrics.symbol_resolution_accuracy.unwrap_or(0.0) * 100.0;
    let acc_without = if sector_eval_without_copy.defined_name_predictions > 0 {
        (sector_eval_without_copy.defined_name_top1_hits as f64
            / sector_eval_without_copy.defined_name_predictions as f64)
            * 100.0
    } else {
        0.0
    };
    println!("   - Top-1 en nombres ya definidos (CON copia de tabla de símbolos):    {:.2}%", acc_with);
    println!("   - Top-1 en nombres ya definidos (SIN tabla de símbolos, solo n-gram): {:.2}%", acc_without);
    println!("   - Ganancia neta por tabla de símbolos: {:+.2} puntos porcentuales", acc_with - acc_without);
    println!("3. Almacenamiento y Carga Perezosa (LRU Cache, cap={}):", args.cache_capacity);
    println!("   - Sectores cargados por cada 1000 tokens predichos: {:.2}", cache_stats.sectors_loaded_per_1000_tokens());
    println!("   - Bytes cargados por cada 1000 tokens predichos:    {:.2} KB ({:.0} bytes)",
             cache_stats.bytes_loaded_per_1000_tokens() / 1024.0, cache_stats.bytes_loaded_per_1000_tokens());
    println!("   - Tasa de acierto de caché LRU (Hit Rate):          {:.2}% ({} hits / {} requests)",
             cache_stats.hit_rate() * 100.0, cache_stats.hits, cache_stats.total_requests);
    println!("   - Memoria residente activa de sectores (real):      {:.2} MB (frente a {:.2} MB del modelo neuronal)",
             sector_lm_mem_mb, starcoder_metrics.as_ref().map_or(f64::NAN, |m| m.memory_mb));
    println!("4. Verificación de Cobertura de Caracteres:");
    println!("   - Sector-LM total caracteres:     {}", sector_lm_metrics.exact_total_chars);
    println!("   - Monolithic BPE total caracteres:{}", bpe_metrics.exact_total_chars);
    assert_eq!(sector_lm_metrics.exact_total_chars, bpe_metrics.exact_total_chars);
    if let Some(sc) = &starcoder_metrics {
        println!("   - StarCoder total caracteres:     {}", sc.exact_total_chars);
        assert_eq!(sector_lm_metrics.exact_total_chars, sc.exact_total_chars);
        println!("   - VERIFICADO: Los 3 modelos evaluaron exactamente {} caracteres.", total_test_chars);
    } else {
        println!("   - (--only-file: sin StarCoder; Sector-LM y BPE coinciden en {} caracteres)", total_test_chars);
    }
    println!("---------------------------------------------------------------------------------------------------------");

    // Save summary to disk
    let summary = json!({
        "options": {"price_sector": args.price_sector, "format": args.format, "spelling": args.spelling, "sectors": args.sectors, "unk_singletons": args.unk_singletons, "text_bpe": args.text_bpe, "spelling_frequency": args.spelling_frequency, "history": format!("{:?}", args.history).to_lowercase()},
        "text_bpe": {"vocab_size": text_tokenizer.as_ref().map_or(0, TextBpe::vocab_size), "training_content_spans": text_training_spans, "sector_id": if args.text_bpe { Some(SectorKind::TextPiece.id()) } else { None::<u8> }},
        "text_type_metrics": {"sector_lm": sector_type_metrics, "monolithic_bpe": bpe_type_metrics},
        "unseen_spelling_nats_per_char": sector_eval_accum.unseen_spelling_nats / sector_eval_accum.unseen_spelling_chars.max(1) as f64,
        "unseen_spelling_characters": sector_eval_accum.unseen_spelling_chars,
        "unseen_token_fraction": sector_eval_accum.per_sector_oov.iter().sum::<u64>() as f64 / sector_eval_accum.total_predictions.max(1) as f64,
        "unk_diagnostics": sector_models.iter().enumerate().map(|(sid, m)| json!({
            "sector_id": sid,
            "singleton_types": m.singleton_count(),
            "training_tokens": m.total_tokens,
            "singleton_token_fraction": m.singleton_count() as f64 / m.total_tokens.max(1) as f64,
            "p_unk_unigram": m.p_continuation(sector_lm::model::ngram::UNK_ID),
        })).collect::<Vec<_>>(),
        "exact_total_chars": total_test_chars,
        "test_files_count": test_items.len(),
        "total_eval_time_secs": total_eval_time_secs,
        "sector_lm": sector_lm_metrics,
        "monolithic_bpe": bpe_metrics,
        "tiny_starcoder_py": starcoder_metrics,
        "symbol_binding_gain_points": acc_with - acc_without,
        "viterbi_disambiguation_accuracy": viterbi_acc,
        "lru_cache_hit_rate": cache_stats.hit_rate(),
        "sectors_loaded_per_1000_tokens": cache_stats.sectors_loaded_per_1000_tokens(),
        "bytes_loaded_per_1000_tokens": cache_stats.bytes_loaded_per_1000_tokens(),
        "slowest_files": sorted_by_time.iter().take(5).map(|f| json!({
            "file_index": f.file_index,
            "tokens": f.token_count,
            "chars": f.exact_char_count,
            "elapsed_seconds": f.elapsed_secs,
        })).collect::<Vec<_>>(),
    });
    if args.only_file.is_some() {
        println!("--only-file: resumen JSON no escrito (prueba de rendimiento).");
    } else {
        if let Some(parent) = Path::new(&args.output_summary).parent() { std::fs::create_dir_all(parent).expect("create summary output directory"); }
        std::fs::write(&args.output_summary, serde_json::to_string_pretty(&summary).unwrap()).expect("write summary JSON");
        println!("Results summary written to '{}'", args.output_summary);
    }
}
