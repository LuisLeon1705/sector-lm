use sector_lm::sector::parser::PythonAstParser;
use sector_lm::sector::{SectorKind, NUM_SECTORS};
use sector_lm::symbol::table::SymbolTable;
use sector_lm::storage::sector_file::{SectorFileData, SectorFileStorage, SectorNgramCounts};
use sector_lm::storage::cache::SectorLruCache;
use sector_lm::model::hmm::SectorHmm;
use sector_lm::data::{DatasetItem, verify_no_leakage};
use sector_lm::sector::domain::remap_tokens_to_domain;
use std::sync::Arc;

#[test]
fn test_viterbi_disambiguation_ambiguous_token() {
    let mut hmm = SectorHmm::new();
    let kw = SectorKind::Keyword.id();
    let def_name = SectorKind::DefinedName.id();
    let used_name = SectorKind::UsedName.id();
    let delim = SectorKind::Delimiter.id();

    // Training sequences:
    // In context "def X": X is DefinedName
    // In context "print(X)": X is UsedName
    let mut training = Vec::new();
    for _ in 0..100 {
        training.push(vec![
            (kw, "def".to_string()),
            (def_name, "Animal".to_string()),
            (delim, ":".to_string()),
        ]);
        training.push(vec![
            (used_name, "print".to_string()),
            (delim, "(".to_string()),
            (used_name, "Animal".to_string()),
            (delim, ")".to_string()),
        ]);
    }
    hmm.fit(&training);

    // Test sequence 1: "def Animal :" -> Animal MUST resolve to DefinedName
    let pred1 = hmm.decode_viterbi(&["def", "Animal", ":"]);
    assert_eq!(pred1[1], def_name, "Viterbi must assign DefinedName to Animal after def");

    // Test sequence 2: "print ( Animal )" -> Animal MUST resolve to UsedName
    let pred2 = hmm.decode_viterbi(&["print", "(", "Animal", ")"]);
    assert_eq!(pred2[2], used_name, "Viterbi must assign UsedName to Animal in function call expression");
}

#[test]
fn test_def_animal_creates_symbol_and_subsequent_use_loads_only_that_sector() {
    let temp_dir = std::env::temp_dir().join(format!("test_animal_lazy_{}", std::process::id()));
    let storage = Arc::new(SectorFileStorage::new(&temp_dir).unwrap());

    // Initialize all sector files on disk
    for sec_id in 0..NUM_SECTORS as u8 {
        let dummy = SectorFileData {
            sector_id: sec_id,
            sector_name: format!("Sector_{:02}", sec_id),
            vocab: vec!["item".to_string()],
            ngram_counts: SectorNgramCounts::default(),
            total_tokens_observed: 1,
        };
        storage.save_sector(&dummy).unwrap();
    }

    let mut cache = SectorLruCache::new(storage, 5);
    let mut sym_table = SymbolTable::new();
    let mut parser = PythonAstParser::new();

    let code = r#"
class Animal:
    pass

x = Animal()
"#;

    let tokens = parser.parse_code(code).expect("Valid python code");

    // 1. Traverse tokens and register definitions into symbol table
    for tok in &tokens {
        if tok.is_definition {
            sym_table.define(
                &tok.text,
                tok.sector.id(),
                tok.condition,
                tok.byte_range.0,
            );
        }
    }

    // Verify Animal was registered in symbol sector 07
    let resolved = sym_table.resolve("Animal").expect("Animal must be in symbol table");
    assert_eq!(resolved.target_sector, SectorKind::DefinedName.id());
    assert_eq!(resolved.address_token.sector, SectorKind::Symbol.id());

    // At this moment, cache should have NO sectors loaded yet (lazy loading)
    assert_eq!(cache.loaded_sector_ids().len(), 0, "No sectors loaded before resolution");

    // 2. When 'Animal' is used later, resolve it and load ONLY its target sector
    let target_sec = resolved.target_sector;
    let sector_data = cache.get_or_load(target_sec).expect("Sector load must succeed");
    assert_eq!(sector_data.sector_id, target_sec);

    // Verify ONLY that sector is loaded in memory!
    let loaded = cache.loaded_sector_ids();
    assert_eq!(loaded, vec![target_sec], "Lazy loading must touch ONLY the referenced sector");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_lazy_loading_does_not_touch_unneeded_sectors() {
    let temp_dir = std::env::temp_dir().join(format!("test_lazy_untouched_{}", std::process::id()));
    let storage = Arc::new(SectorFileStorage::new(&temp_dir).unwrap());

    // Create 13 sectors
    for sec_id in 0..NUM_SECTORS as u8 {
        let dummy = SectorFileData {
            sector_id: sec_id,
            sector_name: format!("Sector_{:02}", sec_id),
            vocab: vec!["test".to_string()],
            ngram_counts: SectorNgramCounts::default(),
            total_tokens_observed: 1,
        };
        storage.save_sector(&dummy).unwrap();
    }

    let mut cache = SectorLruCache::new(storage, 4);

    // Predict sequence requiring only Keyword (1) and Delimiter (11)
    let needed_sectors = vec![SectorKind::Keyword.id(), SectorKind::Delimiter.id()];
    for &sec in &needed_sectors {
        let _ = cache.get_or_load(sec).unwrap();
    }

    let loaded = cache.loaded_sector_ids();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains(&SectorKind::Keyword.id()));
    assert!(loaded.contains(&SectorKind::Delimiter.id()));

    // Verify unneeded sectors (e.g. NumericLiteral 4, StringLiteral 5, Comment 8) were never touched
    assert!(!cache.is_loaded(SectorKind::NumericLiteral.id()));
    assert!(!cache.is_loaded(SectorKind::StringLiteral.id()));
    assert!(!cache.is_loaded(SectorKind::Comment.id()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_no_test_file_leaks_into_training() {
    let train_files = vec![
        DatasetItem::from_content("def foo():\n    return 1\n".to_string(), None),
        DatasetItem::from_content("x = [i for i in range(10)]\n".to_string(), None),
    ];

    let test_files = vec![
        DatasetItem::from_content("def bar():\n    return 2\n".to_string(), None),
    ];

    let (is_clean, leaked) = verify_no_leakage(&train_files, &test_files);
    assert!(is_clean);
    assert!(leaked.is_empty());
}

#[test]
fn domain_mode_loads_fewer_sectors_than_syntactic_mode() {
    let temp_dir = std::env::temp_dir().join(format!("test_domain_cache_{}", std::process::id()));
    let storage = Arc::new(SectorFileStorage::new(&temp_dir).unwrap());
    for sec_id in 0..NUM_SECTORS as u8 {
        storage.save_sector(&SectorFileData {
            sector_id: sec_id,
            sector_name: format!("Sector_{sec_id:02}"),
            vocab: vec!["x".into()],
            ngram_counts: SectorNgramCounts::default(),
            total_tokens_observed: 1,
        }).unwrap();
    }
    let mut parser = PythonAstParser::new();
    let syntax = parser.parse_code_with_format("import torch\ndef f(x):\n    return x + 1\n").unwrap();
    let domain = remap_tokens_to_domain(&syntax).1;
    let mut syntax_cache = SectorLruCache::new(Arc::clone(&storage), 4);
    let mut domain_cache = SectorLruCache::new(storage, 4);
    for tokens in [&syntax, &domain] {
        let cache = if std::ptr::eq(tokens, &syntax) { &mut syntax_cache } else { &mut domain_cache };
        for token in tokens.iter().skip(1) { cache.record_access(token.sector.id()).unwrap(); }
        cache.record_predicted_tokens(tokens.len().saturating_sub(1) as u64);
    }
    assert!(domain_cache.stats().sectors_loaded < syntax_cache.stats().sectors_loaded,
        "domain={} syntactic={}", domain_cache.stats().sectors_loaded, syntax_cache.stats().sectors_loaded);
    std::fs::remove_dir_all(&temp_dir).unwrap();
}

#[test]
fn test_saving_and_loading_sectors_gives_identical_parameters() {
    let temp_dir = std::env::temp_dir().join(format!("test_save_load_exact_{}", std::process::id()));
    let storage = SectorFileStorage::new(&temp_dir).unwrap();

    let mut counts = SectorNgramCounts::default();
    counts.unigrams.insert(10, 500);
    counts.unigrams.insert(25, 300);
    counts.bigrams.insert((10, 25), 180);
    counts.trigrams.insert((10, 25, 10), 45);
    counts.total_unigrams = 800;

    let original = SectorFileData {
        sector_id: SectorKind::Operator.id(),
        sector_name: "Operator".to_string(),
        vocab: vec!["+".to_string(), "-".to_string(), "==".to_string()],
        ngram_counts: counts,
        total_tokens_observed: 800,
    };

    let bytes_written = storage.save_sector(&original).unwrap();
    assert!(bytes_written > 0);

    let (loaded, bytes_mmapped) = storage.load_sector_mmap(original.sector_id).unwrap();
    assert_eq!(bytes_written, bytes_mmapped);
    assert_eq!(original.sector_id, loaded.sector_id);
    assert_eq!(original.vocab, loaded.vocab);
    assert_eq!(original.ngram_counts, loaded.ngram_counts);
    assert_eq!(original.total_tokens_observed, loaded.total_tokens_observed);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
