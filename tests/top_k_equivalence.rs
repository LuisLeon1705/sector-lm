use sector_lm::data::load_dataset_jsonl;
use sector_lm::model::ngram::SectorKneserNeyModel;
use sector_lm::sector::parser::PythonAstParser;
use sector_lm::sector::SectorKind;
use sector_lm::symbol::table::SymbolTable;
use std::path::Path;

#[test]
fn test_top_k_fast_vs_exhaustive_equivalence_on_validation_data() {
    let valid_path = "data/valid.jsonl";
    if !Path::new(valid_path).exists() {
        panic!(
            "FATAL: Required test fixture '{}' is missing! Run 'python scripts/prepare_data.py' first.",
            valid_path
        );
    }

    let valid_items = load_dataset_jsonl(valid_path).expect("Failed to load valid.jsonl");
    assert!(!valid_items.is_empty(), "Validation dataset must not be empty");

    let mut parser = PythonAstParser::new();
    let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());

    // Train on the first 35 validation files to establish a rich, realistic vocabulary with high-fanout tokens
    for item in valid_items.iter().take(35) {
        if let Some(tokens) = parser.parse_code(&item.content) {
            let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
            model.train(&texts);
        }
    }
    model.recompute_continuation_counts();
    assert!(model.vocab.len() >= 100, "Vocabulary must be sufficiently large for a meaningful test");

    let mut sym_table = SymbolTable::new();
    sym_table.define("active_var_1", SectorKind::DefinedName.id(), 'G', 0);
    sym_table.define("custom_helper", SectorKind::DefinedName.id(), 'F', 10);
    sym_table.define("AnimalModel", SectorKind::DefinedName.id(), 'C', 20);

    let k = 5;

    // A. Explicitly test high-fanout contexts (such as after 'self', '.', 'self .')
    let high_fanout_contexts: Vec<Vec<&str>> = vec![
        vec!["self"],
        vec!["."],
        vec!["self", "."],
        vec!["os", "."],
        vec!["data", "."],
        vec!["sys", "."],
        vec!["None"],
        vec!["x"],
        vec!["val", "="],
    ];

    println!("\nTesting {} high-fanout contexts...", high_fanout_contexts.len());
    for ctx in &high_fanout_contexts {
        for &copy_weight in &[0.0, 0.4] {
            let opt_sym = if copy_weight > 0.0 { Some(&sym_table) } else { None };
            let fast = model.top_k(ctx, k, opt_sym, copy_weight);
            let exhaustive = model.top_k_exhaustive(ctx, k, opt_sym, copy_weight);

            assert_eq!(
                fast.len(),
                exhaustive.len(),
                "High-fanout length mismatch for context {:?}",
                ctx
            );

            for rank in 0..fast.len() {
                assert_eq!(
                    fast[rank].0, exhaustive[rank].0,
                    "High-fanout token mismatch at rank {} for context {:?}: fast '{}' vs exhaustive '{}'",
                    rank, ctx, fast[rank].0, exhaustive[rank].0
                );
                let diff = (fast[rank].1 - exhaustive[rank].1).abs();
                assert!(
                    diff < 1e-9,
                    "High-fanout probability mismatch at rank {} for context {:?}: |{} - {}| = {} >= 1e-9",
                    rank, ctx, fast[rank].1, exhaustive[rank].1, diff
                );
            }
        }
    }
    println!("High-fanout contexts passed successfully!");

    // B. Test at least 200 real contexts extracted from subsequent validation files
    let mut tested_contexts_count = 0;
    'outer: for item in valid_items.iter().skip(35) {
        if let Some(tokens) = parser.parse_code(&item.content) {
            let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
            for i in 1..texts.len() {
                // Alternating between bigram, trigram, and unknown tokens in context
                let history_slice: &[&str] = if i >= 2 && tested_contexts_count % 2 == 0 {
                    &texts[i - 2..i]
                } else {
                    &texts[i - 1..i]
                };

                // Alternating copy mechanism
                let use_copy = tested_contexts_count % 3 == 0;
                let opt_sym = if use_copy { Some(&sym_table) } else { None };
                let copy_weight = if use_copy { 0.4 } else { 0.0 };

                let fast_result = model.top_k(history_slice, k, opt_sym, copy_weight);
                let exhaustive_result = model.top_k_exhaustive(history_slice, k, opt_sym, copy_weight);

                assert_eq!(
                    fast_result.len(),
                    exhaustive_result.len(),
                    "Mismatch in top-k length for context {:?}",
                    history_slice
                );

                for rank in 0..fast_result.len() {
                    let (ref fast_tok, fast_prob) = fast_result[rank];
                    let (ref exh_tok, exh_prob) = exhaustive_result[rank];

                    assert_eq!(
                        fast_tok, exh_tok,
                        "Token mismatch at rank {} for context {:?}: fast '{}' vs exhaustive '{}'",
                        rank, history_slice, fast_tok, exh_tok
                    );

                    let prob_diff = (fast_prob - exh_prob).abs();
                    assert!(
                        prob_diff < 1e-9,
                        "Probability mismatch at rank {} for context {:?}: |{} - {}| = {} >= 1e-9",
                        rank, history_slice, fast_prob, exh_prob, prob_diff
                    );
                }

                tested_contexts_count += 1;
                if tested_contexts_count >= 250 {
                    break 'outer;
                }
            }
        }
    }

    assert!(
        tested_contexts_count >= 200,
        "Must verify at least 200 real contexts (tested: {})",
        tested_contexts_count
    );
    println!(
        "VERIFIED: High-fanout and {} real validation contexts tested. Fast top-k with early stopping is strictly identical to exhaustive (prob diff < 1e-9).",
        tested_contexts_count
    );
}

fn assert_same(label: &str, ctx: &[&str], fast: &[(String, f64)], exh: &[(String, f64)]) {
    assert_eq!(fast.len(), exh.len(), "{label}: length mismatch for {ctx:?}");
    for (rank, (f, e)) in fast.iter().zip(exh.iter()).enumerate() {
        assert_eq!(f.0, e.0, "{label}: token mismatch at rank {rank} for {ctx:?}: '{}' vs '{}'", f.0, e.0);
        assert!(
            (f.1 - e.1).abs() < 1e-9,
            "{label}: probability mismatch at rank {rank} for {ctx:?}: {} vs {}",
            f.1,
            e.1
        );
    }
}

/// Both outputs of `top_k_dual` (copy_weight > 0, hundreds of active symbols) must match
/// the exhaustive reference of their own variant.
#[test]
fn test_top_k_dual_both_variants_match_exhaustive_with_many_symbols() {
    let valid_path = "data/valid.jsonl";
    if !Path::new(valid_path).exists() {
        panic!(
            "FATAL: Required test fixture '{}' is missing! Run 'python scripts/prepare_data.py' first.",
            valid_path
        );
    }
    let valid_items = load_dataset_jsonl(valid_path).expect("Failed to load valid.jsonl");
    let mut parser = PythonAstParser::new();
    let mut model = SectorKneserNeyModel::new(SectorKind::UsedName.id());
    let mut parsed: Vec<Vec<String>> = Vec::new();
    for item in valid_items.iter().take(60) {
        if let Some(tokens) = parser.parse_code(&item.content) {
            parsed.push(tokens.iter().map(|t| t.text.clone()).collect());
        }
    }
    for (i, texts) in parsed.iter().enumerate() {
        if i < 35 {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            model.train(&refs);
        }
    }
    model.recompute_continuation_counts();

    // Hundreds of active symbols: half are in-vocabulary identifiers, half are not
    let mut sym_table = SymbolTable::new();
    let in_vocab: Vec<String> = model
        .vocab
        .iter()
        .filter(|t| t.chars().all(|c| c.is_alphanumeric() || c == '_') && t.len() > 2)
        .step_by(5)
        .take(200)
        .cloned()
        .collect();
    assert!(in_vocab.len() >= 100, "need many in-vocabulary symbols, got {}", in_vocab.len());
    for name in &in_vocab {
        sym_table.define(name, SectorKind::DefinedName.id(), 'G', 0);
    }
    for i in 0..200 {
        sym_table.define(&format!("zz_unseen_symbol_{i}"), SectorKind::DefinedName.id(), 'G', 0);
    }
    assert!(sym_table.active_ids().len() >= 300);

    let cw = 0.4;
    let k = 5;
    let checked = std::cell::Cell::new(0usize);
    let check = |ctx: &[&str]| {
        let (with_copy, no_copy) = model.top_k_dual(ctx, k, Some(&sym_table), cw);
        let exh_copy = model.top_k_exhaustive(ctx, k, Some(&sym_table), cw);
        let exh_base = model.top_k_exhaustive(ctx, k, None, 0.0);
        assert_same("with copy", ctx, &with_copy, &exh_copy);
        assert_same("without copy", ctx, &no_copy, &exh_base);
        checked.set(checked.get() + 1);
    };

    let high_fanout: Vec<Vec<&str>> = vec![
        vec!["self"],
        vec!["."],
        vec!["self", "."],
        vec!["os", "."],
        vec!["None"],
        vec!["="],
        vec!["x", "="],
        vec![],
    ];
    for ctx in &high_fanout {
        check(ctx);
    }
    let high_fanout_count = checked.get();

    'outer: for texts in parsed.iter().skip(35) {
        for i in 1..texts.len() {
            let start = if i >= 2 && checked.get().is_multiple_of(2) { i - 2 } else { i - 1 };
            let ctx: Vec<&str> = texts[start..i].iter().map(|s| s.as_str()).collect();
            check(&ctx);
            if checked.get() >= high_fanout_count + 300 {
                break 'outer;
            }
        }
    }
    assert!(checked.get() >= high_fanout_count + 300, "too few real contexts checked: {}", checked.get());
}
