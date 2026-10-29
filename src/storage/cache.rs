use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::Arc;
use crate::storage::sector_file::{SectorFileData, SectorFileStorage};

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub total_requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub sectors_loaded: u64,
    pub bytes_loaded: u64,
    pub tokens_predicted: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.hits as f64) / (self.total_requests as f64)
        }
    }

    pub fn sectors_loaded_per_1000_tokens(&self) -> f64 {
        if self.tokens_predicted == 0 {
            0.0
        } else {
            (self.sectors_loaded as f64) * 1000.0 / (self.tokens_predicted as f64)
        }
    }

    pub fn bytes_loaded_per_1000_tokens(&self) -> f64 {
        if self.tokens_predicted == 0 {
            0.0
        } else {
            (self.bytes_loaded as f64) * 1000.0 / (self.tokens_predicted as f64)
        }
    }
}

pub struct SectorLruCache {
    storage: Arc<SectorFileStorage>,
    capacity: usize,
    map: HashMap<u8, SectorFileData>,
    sector_sizes: HashMap<u8, usize>,
    order: VecDeque<u8>,
    stats: CacheStats,
}

impl SectorLruCache {
    pub fn new(storage: Arc<SectorFileStorage>, capacity: usize) -> Self {
        assert!(capacity > 0, "LRU cache capacity must be > 0");
        Self {
            storage,
            capacity,
            map: HashMap::new(),
            sector_sizes: HashMap::new(),
            order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn get_or_load(&mut self, sector_id: u8) -> io::Result<&SectorFileData> {
        self.stats.total_requests += 1;

        if self.map.contains_key(&sector_id) {
            self.stats.hits += 1;
            // Move to back (most recently used)
            if let Some(pos) = self.order.iter().position(|&x| x == sector_id) {
                self.order.remove(pos);
            }
            self.order.push_back(sector_id);
            return Ok(self.map.get(&sector_id).unwrap());
        }

        // Cache miss: load from disk via mmap
        self.stats.misses += 1;
        let (data, bytes) = self.storage.load_sector_mmap(sector_id)?;
        self.stats.sectors_loaded += 1;
        self.stats.bytes_loaded += bytes as u64;

        if self.map.len() >= self.capacity {
            if let Some(evicted_id) = self.order.pop_front() {
                self.map.remove(&evicted_id);
                self.sector_sizes.remove(&evicted_id);
            }
        }

        self.order.push_back(sector_id);
        self.sector_sizes.insert(sector_id, bytes);
        self.map.insert(sector_id, data);
        Ok(self.map.get(&sector_id).unwrap())
    }

    /// Same LRU bookkeeping and statistics as get_or_load, but without reading or
    /// deserializing the file: a miss counts the real file size from disk metadata.
    /// Used to replay recorded sector accesses (the evaluation predicts from in-memory
    /// models, so only the access pattern is needed). Do not mix with get_or_load
    /// on the same cache.
    pub fn record_access(&mut self, sector_id: u8) -> io::Result<()> {
        self.stats.total_requests += 1;
        if self.sector_sizes.contains_key(&sector_id) {
            self.stats.hits += 1;
            if let Some(pos) = self.order.iter().position(|&x| x == sector_id) {
                self.order.remove(pos);
            }
            self.order.push_back(sector_id);
            return Ok(());
        }
        self.stats.misses += 1;
        let bytes = self.storage.sector_file_len(sector_id)?;
        self.stats.sectors_loaded += 1;
        self.stats.bytes_loaded += bytes as u64;
        if self.sector_sizes.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.sector_sizes.remove(&evicted);
            }
        }
        self.order.push_back(sector_id);
        self.sector_sizes.insert(sector_id, bytes);
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        self.sector_sizes.values().sum()
    }

    pub fn resident_memory_mb(&self) -> f64 {
        self.resident_bytes() as f64 / (1024.0 * 1024.0)
    }

    pub fn is_loaded(&self, sector_id: u8) -> bool {
        self.map.contains_key(&sector_id)
    }

    pub fn loaded_sector_ids(&self) -> Vec<u8> {
        self.map.keys().copied().collect()
    }

    pub fn record_predicted_tokens(&mut self, count: u64) {
        self.stats.tokens_predicted += count;
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::sector_file::SectorNgramCounts;

    #[test]
    fn test_lru_cache_behavior_and_eviction() {
        let temp_dir = std::env::temp_dir().join(format!("lru_test_{}", std::process::id()));
        let storage = Arc::new(SectorFileStorage::new(&temp_dir).unwrap());

        // Create 3 dummy sector files: 1, 2, 3
        for id in 1..=3 {
            let data = SectorFileData {
                sector_id: id,
                sector_name: format!("Sector_{}", id),
                vocab: vec!["tok".to_string()],
                ngram_counts: SectorNgramCounts::default(),
                total_tokens_observed: 10,
            };
            storage.save_sector(&data).unwrap();
        }

        // Create cache with capacity 2
        let mut cache = SectorLruCache::new(storage, 2);

        // Load sector 1
        assert!(!cache.is_loaded(1));
        let s1 = cache.get_or_load(1).unwrap();
        assert_eq!(s1.sector_id, 1);
        assert!(cache.is_loaded(1));
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Access sector 1 again -> hit!
        let _ = cache.get_or_load(1).unwrap();
        assert_eq!(cache.stats().hits, 1);

        // Load sector 2 -> miss, now 1 and 2 in cache
        let _ = cache.get_or_load(2).unwrap();
        assert!(cache.is_loaded(1));
        assert!(cache.is_loaded(2));

        // Load sector 3 -> capacity 2 reached, sector 1 was accessed before 2, so sector 1 should be evicted!
        let _ = cache.get_or_load(3).unwrap();
        assert!(cache.is_loaded(3));
        assert!(cache.is_loaded(2));
        assert!(!cache.is_loaded(1), "Sector 1 should have been evicted by LRU");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_record_access_matches_real_loading_stats() {
        let temp_dir = std::env::temp_dir().join(format!("lru_replay_test_{}", std::process::id()));
        let storage = Arc::new(SectorFileStorage::new(&temp_dir).unwrap());
        for id in 0..5u8 {
            let data = SectorFileData {
                sector_id: id,
                sector_name: format!("Sector_{}", id),
                vocab: (0..(id as usize * 50 + 1)).map(|i| format!("tok{i}")).collect(),
                ngram_counts: SectorNgramCounts::default(),
                total_tokens_observed: 10,
            };
            storage.save_sector(&data).unwrap();
        }
        let seq = [0u8, 1, 0, 2, 3, 4, 0, 0, 1, 2, 3, 3, 4, 1, 0];
        let mut real = SectorLruCache::new(Arc::clone(&storage), 3);
        let mut replay = SectorLruCache::new(Arc::clone(&storage), 3);
        for &s in &seq {
            real.get_or_load(s).unwrap();
            replay.record_access(s).unwrap();
        }
        let (a, b) = (real.stats(), replay.stats());
        assert_eq!(
            (a.total_requests, a.hits, a.misses, a.sectors_loaded, a.bytes_loaded),
            (b.total_requests, b.hits, b.misses, b.sectors_loaded, b.bytes_loaded)
        );
        assert!(a.misses > 5, "sequence must force evictions");
        assert_eq!(real.resident_bytes(), replay.resident_bytes());
        let mut ra = real.loaded_sector_ids();
        ra.sort();
        let mut rb: Vec<u8> = replay.sector_sizes.keys().copied().collect();
        rb.sort();
        assert_eq!(ra, rb);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}