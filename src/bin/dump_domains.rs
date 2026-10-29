use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use serde_json::{json, Map, Value};
use sector_lm::data::load_dataset_jsonl;
use sector_lm::sector::domain::{classify_file_domain, DomainGroup, NUM_DOMAIN_SECTORS};
use sector_lm::sector::parser::PythonAstParser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value="data/train.jsonl")] train_file: PathBuf,
    #[arg(long, default_value="data/valid.jsonl")] valid_file: PathBuf,
    #[arg(long, default_value="data/test.jsonl")] test_file: PathBuf,
    #[arg(long, default_value="data/round5_domains")] output_dir: PathBuf,
}

fn process(path: &PathBuf) -> (Map<String, Value>, [u64; NUM_DOMAIN_SECTORS], [u64; NUM_DOMAIN_SECTORS]) {
    let docs = load_dataset_jsonl(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut mapping = Map::new();
    let mut files = [0u64; NUM_DOMAIN_SECTORS];
    let mut tokens = [0u64; NUM_DOMAIN_SECTORS];
    let start = Instant::now();
    let report_every = (docs.len() / 10).max(1);
    for (idx, item) in docs.iter().enumerate() {
        let mut parser = PythonAstParser::new();
        let parsed = parser.parse_code(&item.content).unwrap_or_default();
        let domain = classify_file_domain(&parsed);
        let token_count = parsed.len().max(1) as u64;
        let id = domain.id() as usize;
        files[id] += 1;
        tokens[id] += token_count;
        mapping.insert(idx.to_string(), json!(domain.id()));
        if (idx + 1) % report_every == 0 || idx + 1 == docs.len() {
            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            let files_per_sec = (idx + 1) as f64 / elapsed;
            let eta = (docs.len() - idx - 1) as f64 / files_per_sec;
            eprintln!("[{path:?}] {:>3}% | {}/{} files | {:.0} files/s | {:.0} tokens/s | ETA {:.0}s", (idx+1)*100/docs.len().max(1), idx+1, docs.len(), files_per_sec, tokens.iter().sum::<u64>() as f64/elapsed, eta);
        }
    }
    (mapping, files, tokens)
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("create domain output directory");
    let splits = [("train", args.train_file), ("valid", args.valid_file), ("test", args.test_file)];
    let mut stats: HashMap<String, Value> = HashMap::new();
    let mut aggregate_tokens = [0u64; NUM_DOMAIN_SECTORS];
    for (name, path) in splits {
        let (mapping, files, tokens) = process(&path);
        std::fs::write(args.output_dir.join(format!("{name}_domains.json")), serde_json::to_vec_pretty(&mapping).unwrap()).expect("write mappings");
        let sum: u64 = tokens.iter().sum();
        let rows: Vec<Value> = (0..NUM_DOMAIN_SECTORS).map(|id| {
            if name == "train" { aggregate_tokens[id] += tokens[id]; }
            json!({"id":id,"name":DomainGroup::from_id(id as u8).unwrap().name(),"files":files[id],"tokens":tokens[id],"token_fraction":tokens[id] as f64/sum.max(1) as f64})
        }).collect();
        stats.insert(name.to_string(), json!({"files":mapping.len(),"tokens":sum,"domains":rows}));
    }
    let all: u64 = aggregate_tokens.iter().sum();
    let merged: Vec<u8> = (0..NUM_DOMAIN_SECTORS-1).filter(|&id| aggregate_tokens[id] as f64 / (all.max(1) as f64) < 0.03).map(|id|id as u8).collect();
    let merged_stats: Vec<Value> = (0..NUM_DOMAIN_SECTORS).map(|id| {
        let dest = if merged.contains(&(id as u8)) { 12 } else { id };
        json!({"source":id,"destination":dest,"tokens":aggregate_tokens[id]})
    }).collect();
    for split in ["train","valid","test"] {
        let path = args.output_dir.join(format!("{split}_domains.json"));
        let mut map: Map<String, Value> = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for value in map.values_mut() {
            if let Some(id) = value.as_u64() { if merged.contains(&(id as u8)) { *value = json!(12); } }
        }
        std::fs::write(path, serde_json::to_vec_pretty(&map).unwrap()).unwrap();
    }
    let report=json!({"splits":stats,"training_merge_below_3_percent":merged,"domain_mapping_after_merge":merged_stats});
    let report_path=args.output_dir.join("domain_report.json");
    std::fs::write(&report_path,serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{}",serde_json::to_string_pretty(&report).unwrap());
}
