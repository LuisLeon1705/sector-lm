use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

pub const SECTOR_MAGIC: &[u8; 8] = b"SECTLM01";

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectorNgramCounts {
    pub unigrams: HashMap<u32, u32>,
    pub bigrams: HashMap<(u32, u32), u32>,
    pub trigrams: HashMap<(u32, u32, u32), u32>,
    pub total_unigrams: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectorFileData {
    pub sector_id: u8,
    pub sector_name: String,
    pub vocab: Vec<String>,
    pub ngram_counts: SectorNgramCounts,
    pub total_tokens_observed: u64,
}

pub struct SectorFileStorage {
    base_dir: PathBuf,
}

impl SectorFileStorage {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    pub fn sector_path(&self, sector_id: u8) -> PathBuf {
        self.base_dir.join(format!("sector_{:02}.bin", sector_id))
    }

    /// Writes sector data to disk as a binary file prefixed by magic bytes.
    pub fn save_sector(&self, data: &SectorFileData) -> io::Result<usize> {
        let path = self.sector_path(data.sector_id);
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.write_all(SECTOR_MAGIC)?;
        let encoded = bincode::serialize(data).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Serialization error: {}", e))
        })?;
        file.write_all(&encoded)?;
        file.flush()?;

        Ok(SECTOR_MAGIC.len() + encoded.len())
    }

    /// Size in bytes of a sector file on disk (the value load_sector_mmap reports).
    pub fn sector_file_len(&self, sector_id: u8) -> io::Result<usize> {
        Ok(std::fs::metadata(self.sector_path(sector_id))?.len() as usize)
    }

    /// Loads a sector file using memmap2 and deserializes it.
    pub fn load_sector_mmap(&self, sector_id: u8) -> io::Result<(SectorFileData, usize)> {
        let path = self.sector_path(sector_id);
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len() as usize;

        // Safety: We ensure the file is opened read-only and remains unmodified during mapping
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < SECTOR_MAGIC.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Sector file too short for magic bytes",
            ));
        }

        if &mmap[0..SECTOR_MAGIC.len()] != SECTOR_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid sector file magic bytes",
            ));
        }

        let payload = &mmap[SECTOR_MAGIC.len()..];
        let data: SectorFileData = bincode::deserialize(payload).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Deserialization error: {}", e))
        })?;

        Ok((data, file_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_load_exact_roundtrip() {
        let temp_dir = std::env::temp_dir().join(format!("sector_test_{}", std::process::id()));
        let storage = SectorFileStorage::new(&temp_dir).unwrap();

        let mut unigrams = HashMap::new();
        unigrams.insert(1, 100);
        unigrams.insert(2, 42);

        let mut bigrams = HashMap::new();
        bigrams.insert((1, 2), 35);

        let orig = SectorFileData {
            sector_id: 1,
            sector_name: "Keyword".to_string(),
            vocab: vec!["def".to_string(), "class".to_string()],
            ngram_counts: SectorNgramCounts {
                unigrams,
                bigrams,
                trigrams: HashMap::new(),
                total_unigrams: 142,
            },
            total_tokens_observed: 142,
        };

        let bytes_written = storage.save_sector(&orig).unwrap();
        assert!(bytes_written > 0);

        let (loaded, bytes_mmapped) = storage.load_sector_mmap(1).unwrap();
        assert_eq!(bytes_written, bytes_mmapped);
        assert_eq!(orig, loaded, "Loaded parameters must be identical to saved parameters");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
