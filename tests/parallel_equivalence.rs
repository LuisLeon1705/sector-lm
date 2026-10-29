use std::sync::Arc;
use sector_lm::data::load_dataset_jsonl;
use sector_lm::eval::{evaluate_dataset_sector_lm_parallel, evaluate_dataset_sector_lm_sequential};
use sector_lm::model::hmm::SectorHmm;
use sector_lm::model::ngram::SectorKneserNeyModel;
use sector_lm::sector::parser::PythonAstParser;
use sector_lm::sector::NUM_SECTORS;
use sector_lm::storage::cache::SectorLruCache;
use sector_lm::storage::sector_file::{SectorFileData, SectorFileStorage, SectorNgramCounts};

#[test]
fn test_parallel_vs_sequential_evaluation_equivalence_on_10_files() {
    let test_path = "data/test.jsonl";
    if !std::path::Path::new(test_path).exists() {
        eprintln!("Warning: data/test.jsonl does not exist, skipping test");
        return;
    }

    let items = load_dataset_jsonl(test_path).expect("Load test.jsonl");
    let mut parser = PythonAstParser::new();

    let mut test_files = Vec::new();
    for it in items.into_iter().take(10) {
        if let Some(toks) = parser.parse_code(&it.content) {
            test_files.push((it.content, toks));
        }
    }
    assert_eq!(test_files.len(), 10, "Should have exactly 10 valid test files");

    // Build HMM and sector models from these files for test
    let mut hmm = SectorHmm::new();
    let hmm_seqs: Vec<Vec<(u8, String)>> = test_files
        .iter()
        .map(|(_, toks)| toks.iter().map(|t| (t.sector.id(), t.text.clone())).collect())
        .collect();
    hmm.fit(&hmm_seqs);

    let mut sector_models: Vec<SectorKneserNeyModel> = (0..NUM_SECTORS as u8)
        .map(SectorKneserNeyModel::new)
        .collect();

    for (_, toks) in &test_files {
        for t in toks {
            let sid = t.sector.id() as usize;
            sector_models[sid].train(&[&t.text]);
        }
    }
    for m in &mut sector_models {
        m.recompute_continuation_counts();
    }

    // Set up dummy sector files on disk for cache testing
    let temp_dir_seq = std::env::temp_dir().join(format!("par_seq_test_{}", std::process::id()));
    let storage_seq = Arc::new(SectorFileStorage::new(&temp_dir_seq).unwrap());
    for s in 0..NUM_SECTORS as u8 {
        let dummy = SectorFileData {
            sector_id: s,
            sector_name: format!("Sector_{:02}", s),
            vocab: vec!["foo".to_string()],
            ngram_counts: SectorNgramCounts::default(),
            total_tokens_observed: 1,
        };
        storage_seq.save_sector(&dummy).unwrap();
    }

    let temp_dir_par = std::env::temp_dir().join(format!("par_par_test_{}", std::process::id()));
    let storage_par = Arc::new(SectorFileStorage::new(&temp_dir_par).unwrap());
    for s in 0..NUM_SECTORS as u8 {
        let dummy = SectorFileData {
            sector_id: s,
            sector_name: format!("Sector_{:02}", s),
            vocab: vec!["foo".to_string()],
            ngram_counts: SectorNgramCounts::default(),
            total_tokens_observed: 1,
        };
        storage_par.save_sector(&dummy).unwrap();
    }

    let mut cache_seq = SectorLruCache::new(storage_seq, 4);
    let mut cache_par = SectorLruCache::new(storage_par, 4);

    let copy_weight = 0.4;

    // 1. Run Sequential
    let (seq_with_copy, seq_no_copy, seq_v_corr, seq_v_tot) =
        evaluate_dataset_sector_lm_sequential(
            &test_files,
            &hmm,
            &sector_models,
            copy_weight,
            &mut cache_seq,
        );

    // 2. Run Parallel
    let (par_with_copy, par_no_copy, par_v_corr, par_v_tot, _results) =
        evaluate_dataset_sector_lm_parallel(
            &test_files,
            &hmm,
            &sector_models,
            copy_weight,
            &mut cache_par,
            None,
        );

    // 3. Compare Accumlators
    assert_eq!(seq_with_copy.total_predictions, par_with_copy.total_predictions);
    assert_eq!(seq_with_copy.top1_hits, par_with_copy.top1_hits);
    assert_eq!(seq_with_copy.top5_hits, par_with_copy.top5_hits);
    assert_eq!(seq_with_copy.total_chars, par_with_copy.total_chars);
    assert!((seq_with_copy.total_log_prob - par_with_copy.total_log_prob).abs() < 1e-9);

    assert_eq!(seq_no_copy.total_predictions, par_no_copy.total_predictions);
    assert_eq!(seq_no_copy.top1_hits, par_no_copy.top1_hits);
    assert_eq!(seq_no_copy.top5_hits, par_no_copy.top5_hits);
    assert_eq!(seq_no_copy.total_chars, par_no_copy.total_chars);
    assert!((seq_no_copy.total_log_prob - par_no_copy.total_log_prob).abs() < 1e-9);

    assert_eq!(seq_v_corr, par_v_corr);
    assert_eq!(seq_v_tot, par_v_tot);

    // 4. Compare LRU Cache Stats (must be 100% identical)
    let stats_seq = cache_seq.stats();
    let stats_par = cache_par.stats();

    assert_eq!(stats_seq.total_requests, stats_par.total_requests);
    assert_eq!(stats_seq.hits, stats_par.hits);
    assert_eq!(stats_seq.misses, stats_par.misses);
    assert_eq!(stats_seq.sectors_loaded, stats_par.sectors_loaded);
    assert_eq!(stats_seq.bytes_loaded, stats_par.bytes_loaded);
    assert_eq!(stats_seq.tokens_predicted, stats_par.tokens_predicted);

    let _ = std::fs::remove_dir_all(&temp_dir_seq);
    let _ = std::fs::remove_dir_all(&temp_dir_par);

    println!("VERIFIED: Parallel evaluation produces 100% identical metrics and LRU cache statistics to sequential.");
}
