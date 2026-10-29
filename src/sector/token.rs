use std::fmt;
use std::str::FromStr;
use serde::{Deserialize, Serialize};

/// Structured Token representing `SS R C IIIIII`
/// - SS: Sector (00..=99)
/// - R:  Rule (ASCII char: 'T' token, 'R' reference, 'D' definition, 'X' expr, etc.)
/// - C:  Condition / context (ASCII char: 'G' global, 'F' function, 'C' class, 'E' expression, 'S' statement, 'A' arg, etc.)
/// - IIIIII: ID within sector (0..=999_999)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructuredToken {
    pub sector: u8,
    pub rule: char,
    pub condition: char,
    pub id: u32,
}

impl StructuredToken {
    pub fn new(sector: u8, rule: char, condition: char, id: u32) -> Self {
        assert!(sector <= 99, "Sector must be <= 99");
        assert!(id <= 999_999, "ID must be <= 999,999");
        Self {
            sector,
            rule,
            condition,
            id,
        }
    }

    /// Formats as the 10-character canonical representation, e.g., "14XE032032"
    pub fn to_canonical(&self) -> String {
        format!(
            "{:02}{}{}{:06}",
            self.sector, self.rule, self.condition, self.id
        )
    }

    /// Compact 64-bit representation for high-speed cache and storage
    pub fn to_u64(&self) -> u64 {
        ((self.sector as u64) << 48)
            | ((self.rule as u64) << 40)
            | ((self.condition as u64) << 32)
            | (self.id as u64)
    }

    pub fn from_u64(val: u64) -> Self {
        let sector = ((val >> 48) & 0xFF) as u8;
        let rule = ((val >> 40) & 0xFF) as u8 as char;
        let condition = ((val >> 32) & 0xFF) as u8 as char;
        let id = (val & 0xFFFFFFFF) as u32;
        Self {
            sector,
            rule,
            condition,
            id,
        }
    }
}

impl fmt::Display for StructuredToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}{}{}{:06}",
            self.sector, self.rule, self.condition, self.id
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseTokenError(String);

impl fmt::Display for ParseTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid structured token format: {}", self.0)
    }
}

impl std::error::Error for ParseTokenError {}

impl FromStr for StructuredToken {
    type Err = ParseTokenError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() != 10 {
            return Err(ParseTokenError(format!("Expected 10 characters, got {}", s.len())));
        }
        let sector = s[0..2]
            .parse::<u8>()
            .map_err(|_| ParseTokenError(format!("Invalid sector: {}", &s[0..2])))?;
        let rule = s.chars().nth(2).unwrap();
        let condition = s.chars().nth(3).unwrap();
        let id = s[4..10]
            .parse::<u32>()
            .map_err(|_| ParseTokenError(format!("Invalid id: {}", &s[4..10])))?;

        Ok(Self::new(sector, rule, condition, id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_and_parse() {
        let token = StructuredToken::new(14, 'X', 'E', 32032);
        let s = token.to_canonical();
        assert_eq!(s, "14XE032032");

        let parsed: StructuredToken = s.parse().unwrap();
        assert_eq!(parsed, token);
    }

    #[test]
    fn test_u64_packing() {
        let token = StructuredToken::new(7, 'R', 'F', 123);
        let packed = token.to_u64();
        let unpacked = StructuredToken::from_u64(packed);
        assert_eq!(unpacked, token);
        assert_eq!(token.to_canonical(), "07RF000123");
    }
}
