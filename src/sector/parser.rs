use tree_sitter::{Node, Parser};
use std::collections::HashMap;
use crate::sector::{SectorKind, NUM_SECTORS};

#[derive(Debug, Clone)]
pub struct ParsedToken {
    pub text: String,
    pub sector: SectorKind,
    pub rule: char,
    pub condition: char,
    pub is_definition: bool,
    pub is_reference: bool,
    pub byte_range: (usize, usize),
}

#[derive(Default, Debug, Clone)]
pub struct SectorVocabularies {
    // Map: sector_id -> (text -> id, Vec<text>)
    vocabs: Vec<HashMap<String, u32>>,
    rev_vocabs: Vec<Vec<String>>,
}

impl SectorVocabularies {
    pub fn new() -> Self {
        let mut vocabs = Vec::with_capacity(NUM_SECTORS);
        let mut rev_vocabs = Vec::with_capacity(NUM_SECTORS);
        for _ in 0..NUM_SECTORS {
            vocabs.push(HashMap::new());
            rev_vocabs.push(Vec::new());
        }
        Self { vocabs, rev_vocabs }
    }

    pub fn get_or_insert(&mut self, sector_id: u8, text: &str) -> u32 {
        let sid = sector_id as usize;
        if let Some(&id) = self.vocabs[sid].get(text) {
            id
        } else {
            let id = self.rev_vocabs[sid].len() as u32;
            self.vocabs[sid].insert(text.to_string(), id);
            self.rev_vocabs[sid].push(text.to_string());
            id
        }
    }

    pub fn get(&self, sector_id: u8, text: &str) -> Option<u32> {
        let sid = sector_id as usize;
        self.vocabs.get(sid).and_then(|v| v.get(text).copied())
    }

    pub fn get_text(&self, sector_id: u8, id: u32) -> Option<&str> {
        let sid = sector_id as usize;
        self.rev_vocabs
            .get(sid)
            .and_then(|v| v.get(id as usize).map(|s| s.as_str()))
    }

    pub fn sector_vocab_size(&self, sector_id: u8) -> usize {
        self.rev_vocabs.get(sector_id as usize).map_or(0, |v| v.len())
    }
}

/// Splits gap (starting at byte ase of the source) into maximal runs of
/// whitespace / non-whitespace, each one a Format token.
fn push_gap_tokens(gap: &str, base: usize, out: &mut Vec<ParsedToken>) {
    let mut run_start = 0usize;
    let mut run_ws: Option<bool> = None;
    let flush = |s: usize, e: usize, out: &mut Vec<ParsedToken>| {
        if e > s {
            out.push(ParsedToken {
                text: gap[s..e].to_string(),
                sector: SectorKind::Format,
                rule: 'T',
                condition: 'G',
                is_definition: false,
                is_reference: false,
                byte_range: (base + s, base + e),
            });
        }
    };
    for (i, ch) in gap.char_indices() {
        let ws = ch.is_whitespace();
        match run_ws {
            Some(prev) if prev == ws => {}
            Some(_) => {
                flush(run_start, i, out);
                run_start = i;
                run_ws = Some(ws);
            }
            None => {
                run_start = i;
                run_ws = Some(ws);
            }
        }
    }
    flush(run_start, gap.len(), out);
}

fn add_format_tokens(source: &str, leaves: Vec<ParsedToken>) -> Option<Vec<ParsedToken>> {
    let mut out = Vec::with_capacity(leaves.len() * 2);
    let mut cursor = 0usize;
    for t in leaves {
        let raw = source.get(t.byte_range.0..t.byte_range.1)?;
        // Leaf text is the trimmed node text; locate it inside the node range
        let start = t.byte_range.0 + (raw.len() - raw.trim_start().len());
        let end = start + t.text.len();
        if source.get(start..end)? != t.text || start < cursor {
            return None;
        }
        push_gap_tokens(&source[cursor..start], cursor, &mut out);
        out.push(t);
        cursor = end;
    }
    push_gap_tokens(&source[cursor..], cursor, &mut out);
    Some(out)
}

pub struct PythonAstParser {
    parser: Parser,
}

impl Default for PythonAstParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonAstParser {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        let language = tree_sitter_python::language();
        parser
            .set_language(&language)
            .expect("Failed to set Python grammar for tree-sitter");
        Self { parser }
    }

    /// Parses Python source code. Returns None if the source contains syntax errors.
    pub fn parse_code(&mut self, source: &str) -> Option<Vec<ParsedToken>> {
        let tree = self.parser.parse(source, None)?;
        let root = tree.root_node();
        if root.has_error() {
            return None;
        }

        let mut tokens = Vec::new();
        let source_bytes = source.as_bytes();
        Self::collect_tokens(root, source_bytes, &mut tokens, 'G');
        Some(tokens)
    }

    /// Like parse_code, but also emits Format tokens for all text between leaves
    /// (and before the first / after the last), so that concatenating every token's
    /// text reproduces source exactly. Returns None on syntax errors or if the
    /// leaves do not tile the source consistently.
    pub fn parse_code_with_format(&mut self, source: &str) -> Option<Vec<ParsedToken>> {
        let leaves = self.parse_code(source)?;
        add_format_tokens(source, leaves)
    }

    fn collect_tokens(
        node: Node,
        source_bytes: &[u8],
        tokens: &mut Vec<ParsedToken>,
        current_cond: char,
    ) {
        let node_kind = node.kind();
        let mut next_cond = current_cond;

        // Context / Condition derivation
        match node_kind {
            "function_definition" => next_cond = 'F',
            "class_definition" => next_cond = 'C',
            "parameters" => next_cond = 'A',
            "binary_operator" | "boolean_operator" | "comparison_operator" | "call"
                if current_cond != 'F' && current_cond != 'C' => {
                next_cond = 'E';
            }
            _ => {}
        }

        // Check if node is leaf (no named or unnamed children)
        if node.child_count() == 0 {
            let text = &source_bytes[node.start_byte()..node.end_byte()];
            let text_str = match std::str::from_utf8(text) {
                Ok(s) => s.trim(),
                Err(_) => return,
            };

            if text_str.is_empty() {
                return;
            }

            let (sector, rule, is_def, is_ref) = Self::classify_leaf(node, text_str);
            tokens.push(ParsedToken {
                text: text_str.to_string(),
                sector,
                rule,
                condition: next_cond,
                is_definition: is_def,
                is_reference: is_ref,
                byte_range: (node.start_byte(), node.end_byte()),
            });
            return;
        }

        // Special handling for string tokens and comments that might have children (e.g. interpolations)
        if node_kind == "string" || node_kind == "comment" {
            let text = &source_bytes[node.start_byte()..node.end_byte()];
            if let Ok(text_str) = std::str::from_utf8(text) {
                let text_str = text_str.trim();
                if !text_str.is_empty() {
                    let sector = if node_kind == "comment" {
                        SectorKind::Comment
                    } else {
                        SectorKind::StringLiteral
                    };
                    tokens.push(ParsedToken {
                        text: text_str.to_string(),
                        sector,
                        rule: 'T',
                        condition: next_cond,
                        is_definition: false,
                        is_reference: false,
                        byte_range: (node.start_byte(), node.end_byte()),
                    });
                    return;
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_tokens(child, source_bytes, tokens, next_cond);
        }
    }

    fn classify_leaf(node: Node, text: &str) -> (SectorKind, char, bool, bool) {
        let parent = node.parent();
        let parent_kind = parent.map(|p| p.kind()).unwrap_or("");
        let kind = node.kind();

        // 1. Comments
        if kind == "comment" {
            return (SectorKind::Comment, 'T', false, false);
        }

        // 2. Decorators
        if parent_kind == "decorator" || text.starts_with('@') {
            return (SectorKind::Decorator, 'T', false, false);
        }

        // 3. Imports
        if parent_kind == "import_statement"
            || parent_kind == "import_from_statement"
            || parent_kind == "dotted_name"
            || parent_kind == "aliased_import"
        {
            if text == "import" || text == "from" || text == "as" {
                return (SectorKind::Keyword, 'T', false, false);
            }
            return (SectorKind::Import, 'T', false, false);
        }

        // 4. Keywords
        if Self::is_keyword(text) {
            return (SectorKind::Keyword, 'T', false, false);
        }

        // 5. Builtins & Types
        if Self::is_builtin(text) {
            return (SectorKind::BuiltinType, 'T', false, false);
        }

        // 6. Delimiters & Punctuation
        if Self::is_delimiter(text) {
            return (SectorKind::Delimiter, 'T', false, false);
        }

        // 7. Operators
        if Self::is_operator(text) {
            return (SectorKind::Operator, 'T', false, false);
        }

        // 8. Literals (Numbers)
        if kind == "integer" || kind == "float" || text.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '_') {
            return (SectorKind::NumericLiteral, 'T', false, false);
        }

        // 9. Identifiers (Defined vs Used)
        if kind == "identifier" {
            let is_def = Self::is_definition_context(node, parent);
            if is_def {
                // Rule 'D': Definition that registers into symbol table
                return (SectorKind::DefinedName, 'D', true, false);
            } else {
                // Rule 'R': Reference that resolves via symbol table
                return (SectorKind::UsedName, 'R', false, true);
            }
        }

        // Fallback
        (SectorKind::Control, 'T', false, false)
    }

    fn is_definition_context(node: Node, parent: Option<Node>) -> bool {
        let parent = match parent {
            Some(p) => p,
            None => return false,
        };
        let pkind = parent.kind();
        match pkind {
            "function_definition" => {
                // Check if this node is the function name
                parent.child_by_field_name("name") == Some(node)
            }
            "class_definition" => {
                parent.child_by_field_name("name") == Some(node)
            }
            "parameters" | "typed_parameter" | "default_parameter" => true,
            "assignment" => {
                // Left hand side of assignment
                if let Some(left) = parent.child_by_field_name("left") {
                    left.start_byte() <= node.start_byte() && node.end_byte() <= left.end_byte()
                } else {
                    false
                }
            }
            "for_statement" => {
                if let Some(left) = parent.child_by_field_name("left") {
                    left.start_byte() <= node.start_byte() && node.end_byte() <= left.end_byte()
                } else {
                    false
                }
            }
            "with_item" => {
                if let Some(value) = parent.child_by_field_name("alias") {
                    value.start_byte() <= node.start_byte() && node.end_byte() <= value.end_byte()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn is_keyword(text: &str) -> bool {
        matches!(
            text,
            "def" | "class" | "return" | "if" | "elif" | "else" | "for" | "while"
                | "try" | "except" | "finally" | "with" | "as" | "yield" | "raise"
                | "pass" | "break" | "continue" | "lambda" | "assert" | "async"
                | "await" | "global" | "nonlocal" | "del" | "match" | "case"
        )
    }

    fn is_builtin(text: &str) -> bool {
        matches!(
            text,
            "True" | "False" | "None" | "self" | "cls" | "int" | "str" | "float"
                | "bool" | "list" | "dict" | "set" | "tuple" | "len" | "range"
                | "print" | "type" | "isinstance" | "issubclass" | "open" | "super"
                | "enumerate" | "zip" | "map" | "filter" | "any" | "all" | "sum"
        )
    }

    fn is_delimiter(text: &str) -> bool {
        matches!(
            text,
            "(" | ")" | "[" | "]" | "{" | "}" | ":" | "," | ";" | "." | "->" | "..."
        )
    }

    fn is_operator(text: &str) -> bool {
        matches!(
            text,
            "+" | "-" | "*" | "/" | "//" | "%" | "**" | "==" | "!=" | "<" | ">"
                | "<=" | ">=" | "and" | "or" | "not" | "in" | "is" | "&" | "|"
                | "^" | "~" | "<<" | ">>" | "=" | "+=" | "-=" | "*=" | "/="
                | "//=" | "%=" | "**=" | "&=" | "|=" | "^=" | "<<=" | ">>="
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sitter_python_parse() {
        let mut parser = PythonAstParser::new();
        let code = r#"
def animal(name):
    print(name)
"#;
        let tokens = parser.parse_code(code).expect("Should parse valid python");
        assert!(!tokens.is_empty());

        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"def"));
        assert!(texts.contains(&"animal"));
        assert!(texts.contains(&"name"));

        let def_tok = tokens.iter().find(|t| t.text == "def").unwrap();
        assert_eq!(def_tok.sector, SectorKind::Keyword);

        let animal_tok = tokens.iter().find(|t| t.text == "animal").unwrap();
        assert_eq!(animal_tok.sector, SectorKind::DefinedName);
        assert_eq!(animal_tok.rule, 'D');

        let used_name = tokens.iter().rposition(|t| t.text == "name").unwrap();
        assert_eq!(tokens[used_name].sector, SectorKind::UsedName);
        assert_eq!(tokens[used_name].rule, 'R');
    }

    #[test]
    fn test_tree_sitter_invalid_python_discarded() {
        let mut parser = PythonAstParser::new();
        let code = r#"
def invalid_func(
    broken syntax !!!
"#;
        let res = parser.parse_code(code);
        assert!(res.is_none(), "Invalid syntax must be discarded");
    }
}
