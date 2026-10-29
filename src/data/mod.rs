use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetItem {
    pub content: String,
    pub sha256: String,
    pub path: Option<String>,
}

impl DatasetItem {
    pub fn from_content(content: String, path: Option<String>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        Self {
            content,
            sha256: hash,
            path,
        }
    }
}

/// Verifies that no content hash from test appears anywhere in the training set.
pub fn verify_no_leakage(train: &[DatasetItem], test: &[DatasetItem]) -> (bool, Vec<String>) {
    let mut train_hashes = HashSet::new();
    for item in train {
        train_hashes.insert(&item.sha256);
    }

    let mut leaked = Vec::new();
    for item in test {
        if train_hashes.contains(&item.sha256) {
            leaked.push(item.sha256.clone());
        }
    }

    let clean = leaked.is_empty();
    (clean, leaked)
}

pub fn load_dataset_jsonl<P: AsRef<Path>>(path: P) -> io::Result<Vec<DatasetItem>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(item) = serde_json::from_str::<DatasetItem>(&line) {
            items.push(item);
        } else {
            // Alternatively parse {"content": "..."}
            #[derive(Deserialize)]
            struct SimpleDoc {
                content: String,
            }
            if let Ok(simple) = serde_json::from_str::<SimpleDoc>(&line) {
                items.push(DatasetItem::from_content(simple.content, None));
            }
        }
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leakage_detection() {
        let doc1 = DatasetItem::from_content("print('hello world')\n".to_string(), None);
        let doc2 = DatasetItem::from_content("x = 42\n".to_string(), None);
        let doc3 = DatasetItem::from_content("def foo(): pass\n".to_string(), None);

        let train = vec![doc1.clone(), doc2.clone()];
        let test_clean = vec![doc3.clone()];
        let (is_clean, leaked) = verify_no_leakage(&train, &test_clean);
        assert!(is_clean, "No documents should leak");
        assert!(leaked.is_empty());

        let test_leaked = vec![doc2.clone(), doc3.clone()];
        let (is_clean_leaked, leaked_hashes) = verify_no_leakage(&train, &test_leaked);
        assert!(!is_clean_leaked, "Leaked document should be flagged");
        assert_eq!(leaked_hashes.len(), 1);
        assert_eq!(leaked_hashes[0], doc2.sha256);
    }
}
