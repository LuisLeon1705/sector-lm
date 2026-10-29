//! BPE vocabulary dedicated to comment and string contents. Delimiters and prefixes remain
//! ordinary syntax tokens; the BPE pieces preserve source slices for exact reconstruction.
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::TokenizerBuilder;
use crate::sector::parser::ParsedToken;
use crate::sector::SectorKind;

pub struct TextBpe {
    tokenizer: tokenizers::Tokenizer,
}

fn content_range(tok: &ParsedToken) -> Option<(usize, usize, usize)> {
    let s = tok.text.as_str();
    if tok.sector == SectorKind::Comment {
        return s.strip_prefix('#').map(|_| (1, s.len(), 0));
    }
    if tok.sector != SectorKind::StringLiteral { return None; }
    let bytes = s.as_bytes();
    let mut quote_at = 0;
    while quote_at < bytes.len() && bytes[quote_at].is_ascii_alphabetic() && quote_at < 2 { quote_at += 1; }
    if quote_at >= bytes.len() || (bytes[quote_at] != b'\'' && bytes[quote_at] != b'"') { return None; }
    let quote = bytes[quote_at];
    let delimiter_len = if bytes.get(quote_at..quote_at + 3) == Some(&[quote, quote, quote]) { 3 } else { 1 };
    let open_end = quote_at + delimiter_len;
    let close_start = s.len().checked_sub(delimiter_len)?;
    if close_start < open_end || bytes.get(close_start..)? != &bytes[quote_at..open_end] { return None; }
    Some((open_end, close_start, quote_at))
}

pub fn text_contents(tokens: &[ParsedToken]) -> Vec<String> {
    tokens.iter().filter_map(|t| {
        let (start, end, _) = content_range(t)?;
        Some(t.text[start..end].to_string())
    }).filter(|s| !s.is_empty()).collect()
}

impl TextBpe {
    pub fn vocab_size(&self) -> usize { self.tokenizer.get_vocab_size(false) }
    pub fn train(contents: &[String], vocab_size: usize) -> Result<Self, String> {
        let mut trainer = BpeTrainerBuilder::new().show_progress(false).vocab_size(vocab_size.max(300)).min_frequency(2).initial_alphabet(ByteLevel::alphabet()).build();
        let mut tokenizer = TokenizerBuilder::new().with_model(BPE::default()).with_normalizer(None::<tokenizers::normalizers::NFC>).with_pre_tokenizer(Some(ByteLevel::default())).with_post_processor(None::<tokenizers::processors::byte_level::ByteLevel>).with_decoder(None::<tokenizers::decoders::byte_level::ByteLevel>).build().map_err(|e| e.to_string())?;
        tokenizer.train(&mut trainer, contents.iter().map(String::as_str)).map_err(|e| e.to_string())?;
        Ok(Self { tokenizer: tokenizer.into() })
    }

    pub fn tokenize(&self, tokens: &[ParsedToken]) -> Result<Vec<ParsedToken>, String> {
        let mut out = Vec::with_capacity(tokens.len());
        for tok in tokens {
            let Some((start, end, _)) = content_range(tok) else { out.push(tok.clone()); continue; };
            let opening_end = start;
            if opening_end > 0 { out.push(slice_token(tok, 0, opening_end, tok.sector, tok.condition)); }
            let content = &tok.text[start..end];
            if !content.is_empty() {
                let enc = self.tokenizer.encode(content, false).map_err(|e| e.to_string())?;
                let char_to_byte: Vec<usize> = content.char_indices().map(|(i, _)| i).chain(std::iter::once(content.len())).collect();
                let offsets = enc.get_offsets();
                let offsets_are_chars = offsets.iter().map(|&(_, e)| e).max().unwrap_or(0) <= content.chars().count();
                let mut cursor = 0;
                for &(a, b) in offsets {
                    let (a, b) = if offsets_are_chars { (char_to_byte[a], char_to_byte[b]) } else { (a, b) };
                    let a = a.max(cursor);
                    if b <= a { continue; }
                    if a > cursor { out.push(slice_token(tok, start + cursor, start + a, SectorKind::TextPiece, text_condition(tok.sector))); }
                    if b > a {
                        out.push(slice_token(tok, start + a, start + b, SectorKind::TextPiece, text_condition(tok.sector)));
                        cursor = b;
                    }
                }
                if cursor < content.len() { out.push(slice_token(tok, start + cursor, end, SectorKind::TextPiece, text_condition(tok.sector))); }
            }
            if end < tok.text.len() { out.push(slice_token(tok, end, tok.text.len(), tok.sector, tok.condition)); }
        }
        Ok(out)
    }
}

fn text_condition(sector: SectorKind) -> char { if sector == SectorKind::Comment { 'M' } else { 'S' } }

fn slice_token(tok: &ParsedToken, start: usize, end: usize, sector: SectorKind, condition: char) -> ParsedToken {
    ParsedToken {
        text: tok.text[start..end].to_string(), sector, rule: 'T', condition,
        is_definition: false, is_reference: false,
        byte_range: (tok.byte_range.0 + start, tok.byte_range.0 + end),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::parser::PythonAstParser;
    #[test]
    fn bpe_text_pieces_reconstruct_comments_and_prefixed_strings() {
        let src = "# repeated comment words repeated comment words\nx = rf'''héllo {name} repeated words'''\ny = b\"bytes\"\n";
        let mut p = PythonAstParser::new();
        let tokens = p.parse_code_with_format(src).unwrap();
        let content = text_contents(&tokens);
        let model = TextBpe::train(&content, 400).unwrap();
        let split = model.tokenize(&tokens).unwrap();
        assert_eq!(split.iter().map(|t| t.text.as_str()).collect::<String>(), src);
        assert!(split.iter().any(|t| t.sector == SectorKind::TextPiece && t.condition == 'M'));
        assert!(split.iter().any(|t| t.sector == SectorKind::TextPiece && t.condition == 'S'));
    }
}
