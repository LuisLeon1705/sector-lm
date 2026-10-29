use sector_lm::data::load_dataset_jsonl;
use sector_lm::sector::parser::PythonAstParser;
use sector_lm::sector::SectorKind;
use sector_lm::model::text_bpe::{TextBpe, text_contents};

fn require(path: &str) -> Vec<sector_lm::data::DatasetItem> {
    if !std::path::Path::new(path).exists() {
        panic!("FATAL: required fixture '{path}' is missing! Run 'python scripts/prepare_data.py' first.");
    }
    load_dataset_jsonl(path).expect("load dataset")
}

fn check_roundtrip(parser: &mut PythonAstParser, src: &str) -> bool {
    match parser.parse_code_with_format(src) {
        None => false,
        Some(tokens) => {
            let rebuilt: String = tokens.iter().map(|t| t.text.as_str()).collect();
            assert_eq!(rebuilt, src, "concatenation of all tokens must equal the file");
            // every token must sit exactly where the previous one ended
            let mut pos = 0;
            for t in &tokens {
                assert_eq!(&src[pos..pos + t.text.len()], t.text, "token out of place at byte {pos}");
                if t.sector == SectorKind::Format {
                    assert_eq!(t.byte_range, (pos, pos + t.text.len()));
                }
                pos += t.text.len();
            }
            assert_eq!(pos, src.len());            true
        }
    }
}

#[test]
fn test_format_tokens_reconstruct_handcrafted_sources() {
    let mut p = PythonAstParser::new();
    let cases = [
        "def f(x):\n    return x  # trailing comment   \n",
        "\n\n  \nx = 1\r\ny = 2\t\n",
        "if a:\n\tb = 1\n\n\n    # odd indent comment\nc = (1 +\n     2)\n",
        "s = 'abc'  \nt = \"\"\"multi\n   line\"\"\"   \nu = a \\\n    + b\n",
        "class A:\n    '''doc'''\n\n    def m(self): pass\n",
        "",
        "   ",
        "x = 1",
    ];
    for src in cases {
        assert!(check_roundtrip(&mut p, src), "case failed to parse: {src:?}");
    }
}

#[test]
fn test_format_tokens_reconstruct_real_files() {
    let mut p = PythonAstParser::new();
    let mut parsed = 0usize;
    let mut format_chars = 0usize;
    let mut total_chars = 0usize;
    for path in ["data/valid.jsonl", "data/test.jsonl"] {
        for item in require(path) {
            let src = item.content;
            if p.parse_code(&src).is_none() {
                continue;
            }
            let tokens = p
                .parse_code_with_format(&src)
                .unwrap_or_else(|| panic!("leaves do not tile a parseable file in {path}"));
            let rebuilt: String = tokens.iter().map(|t| t.text.as_str()).collect();
            assert_eq!(rebuilt, src, "reconstruction failed in {path}");
            format_chars += tokens
                .iter()
                .filter(|t| t.sector == SectorKind::Format)
                .map(|t| t.text.chars().count())
                .sum::<usize>();
            total_chars += src.chars().count();
            parsed += 1;
        }
    }
    assert!(parsed >= 500, "too few files exercised: {parsed}");
    let share = format_chars as f64 / total_chars as f64;
    println!("format share of characters: {:.2}% over {parsed} files", share * 100.0);
    assert!(share > 0.05, "format sector is suspiciously small: {share}");
}

#[test]
fn test_text_bpe_pieces_reconstruct_comments_and_strings() {
    let cases = [
        "# a comment with repeated words and emoji 😀\nx = 'hello world'\n",
        "r'''triple\nquoted text'''\nf\"prefix {x} body\"\n",
        "#tail",
    ];
    let mut parser = PythonAstParser::new();
    let parsed: Vec<_> = cases.iter().map(|s| parser.parse_code_with_format(s).unwrap()).collect();
    let corpus: Vec<String> = parsed.iter().flat_map(|t| text_contents(t)).collect();
    let bpe = TextBpe::train(&corpus, 400).unwrap();
    for (source, tokens) in cases.iter().zip(parsed.iter()) {
        let split = bpe.tokenize(tokens).unwrap();
        let rebuilt: String = split.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(&rebuilt, source);
    }
}
