//! Shared construction of model training and evaluation contexts.
use crate::sector::parser::ParsedToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum HistoryMode { Sector, Mixed }

pub fn sector_marker(sector_id: u8) -> String { format!("<sec:{sector_id:02}>") }

/// Returns the previous two context tokens in the exact form used by training.
pub fn history_context(tokens: &[ParsedToken], target_index: usize, mode: HistoryMode) -> Vec<String> {
    let target_sector = tokens[target_index].sector.id();
    let start = target_index.saturating_sub(2);
    match mode {
        HistoryMode::Mixed => tokens[start..target_index].iter().map(|t| {
            if t.sector.id() == target_sector { t.text.clone() } else { sector_marker(t.sector.id()) }
        }).collect(),
        HistoryMode::Sector => tokens[..target_index].iter().enumerate()
            .filter(|(_, t)| t.sector.id() == target_sector)
            .map(|(_, t)| t.text.clone()).rev().take(2).collect::<Vec<_>>().into_iter().rev().collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingEvent { pub target_sector: u8, pub history: Vec<String>, pub target: String }

pub fn training_events(tokens: &[ParsedToken], mode: HistoryMode) -> Vec<TrainingEvent> {
    tokens.iter().enumerate().map(|(i, t)| TrainingEvent {
        target_sector: t.sector.id(), history: history_context(tokens, i, mode), target: t.text.clone(),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::SectorKind;
    use crate::sector::parser::PythonAstParser;

    #[test]
    fn training_and_evaluation_contexts_match_for_each_target() {
        let mut parser = PythonAstParser::new();
        let tokens = parser.parse_code_with_format("x = 'hello' # note\ny = 3\n").unwrap();
        for mode in [HistoryMode::Sector, HistoryMode::Mixed] {
            let events = training_events(&tokens, mode);
            assert_eq!(events.len(), tokens.len());
            for (i, (event, target)) in events.iter().zip(&tokens).enumerate() {
                assert_eq!(event.target_sector, target.sector.id());
                assert_eq!(event.target, target.text);
                assert_eq!(event.history, history_context(&tokens, i, mode));
            }
            let mut models: Vec<_> = (0..crate::sector::NUM_SECTORS as u8)
                .map(crate::model::ngram::SectorKneserNeyModel::new).collect();
            if mode == HistoryMode::Sector {
                let mut streams: Vec<Vec<&str>> = vec![Vec::new(); crate::sector::NUM_SECTORS];
                for token in &tokens { streams[token.sector.id() as usize].push(&token.text); }
                for (id, stream) in streams.iter().enumerate() { models[id].train(stream); }
            } else {
                for event in &events {
                    let history: Vec<&str> = event.history.iter().map(String::as_str).collect();
                    models[event.target_sector as usize].train_event(&history, &event.target);
                }
            }
            for event in &events {
                let model = &models[event.target_sector as usize];
                let target = model.get_token_id_or_unk(&event.target);
                match event.history.as_slice() {
                    [] => assert!(model.unigram_counts.get(&target).copied().unwrap_or(0) > 0),
                    [h] => {
                        let h = model.get_token_id_or_unk(h);
                        assert!(model.bigram_counts.get(&(h, target)).copied().unwrap_or(0) > 0,
                            "training did not register history {:?} -> {:?} in {mode:?}", event.history, event.target);
                    }
                    [h1, h2] => {
                        let h1 = model.get_token_id_or_unk(h1);
                        let h2 = model.get_token_id_or_unk(h2);
                        assert!(model.trigram_counts.get(&(h1, h2, target)).copied().unwrap_or(0) > 0,
                            "training did not register history {:?} -> {:?} in {mode:?}", event.history, event.target);
                    }
                    _ => unreachable!("contexts are capped at two tokens"),
                }
            }
        }
        let mixed = training_events(&tokens, HistoryMode::Mixed);
        assert!(mixed.iter().flat_map(|e| &e.history).any(|h| h == &sector_marker(SectorKind::Format.id())));
    }
}
