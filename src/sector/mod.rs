pub mod token;
pub mod parser;
pub mod domain;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SectorKind {
    Control = 0,
    Keyword = 1,
    DefinedName = 2,
    UsedName = 3,
    NumericLiteral = 4,
    StringLiteral = 5,
    Operator = 6,
    Symbol = 7,
    Comment = 8,
    Decorator = 9,
    Import = 10,
    Delimiter = 11,
    BuiltinType = 12,
    /// Text between leaves: spaces, tabs, newlines and indentation (and any other
    /// uncovered text), one token per maximal run.
    Format = 13,
    TextPiece = 14,
}

pub const NUM_SECTORS: usize = 15;

impl SectorKind {
    pub fn id(&self) -> u8 {
        *self as u8
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(SectorKind::Control),
            1 => Some(SectorKind::Keyword),
            2 => Some(SectorKind::DefinedName),
            3 => Some(SectorKind::UsedName),
            4 => Some(SectorKind::NumericLiteral),
            5 => Some(SectorKind::StringLiteral),
            6 => Some(SectorKind::Operator),
            7 => Some(SectorKind::Symbol),
            8 => Some(SectorKind::Comment),
            9 => Some(SectorKind::Decorator),
            10 => Some(SectorKind::Import),
            11 => Some(SectorKind::Delimiter),
            12 => Some(SectorKind::BuiltinType),
            13 => Some(SectorKind::Format),
            14 => Some(SectorKind::TextPiece),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SectorKind::Control => "Control",
            SectorKind::Keyword => "Keyword",
            SectorKind::DefinedName => "DefinedName",
            SectorKind::UsedName => "UsedName",
            SectorKind::NumericLiteral => "NumericLiteral",
            SectorKind::StringLiteral => "StringLiteral",
            SectorKind::Operator => "Operator",
            SectorKind::Symbol => "Symbol",
            SectorKind::Comment => "Comment",
            SectorKind::Decorator => "Decorator",
            SectorKind::Import => "Import",
            SectorKind::Delimiter => "Delimiter",
            SectorKind::BuiltinType => "BuiltinType",
            SectorKind::Format => "Format",
            SectorKind::TextPiece => "TextPiece",
        }
    }
}
