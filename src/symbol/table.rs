use std::collections::HashMap;
use crate::sector::token::StructuredToken;
use crate::sector::SectorKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolEntry {
    pub name: String,
    pub target_sector: u8,
    pub slot: u32,
    pub address_token: StructuredToken,
    pub scope_level: usize,
    pub byte_offset: usize,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    // Stack of lexical scopes (innermost is last)
    scopes: Vec<HashMap<String, SymbolEntry>>,
    // Global list of entries allocated in symbol sector 07
    entries: Vec<SymbolEntry>,
    // Name interner: symbol name <-> dense id
    names: Vec<String>,
    name_ids: HashMap<String, u32>,
    // Ids of names visible from the current scope (union over all open scopes)
    active_vec: Vec<u32>,
    active_pos: HashMap<u32, usize>,
    // Number of open scopes that define each id
    active_refs: HashMap<u32, u32>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    pub const SYMBOL_SECTOR: u8 = SectorKind::Symbol as u8; // 07

    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()], // Start with global scope (level 0)
            entries: Vec::new(),
            names: Vec::new(),
            name_ids: HashMap::new(),
            active_vec: Vec::new(),
            active_pos: HashMap::new(),
            active_refs: HashMap::new(),
        }
    }

    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.name_ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.name_ids.insert(name.to_string(), id);
        id
    }

    fn activate(&mut self, id: u32) {
        let r = self.active_refs.entry(id).or_insert(0);
        *r += 1;
        if *r == 1 {
            self.active_pos.insert(id, self.active_vec.len());
            self.active_vec.push(id);
        }
    }

    fn deactivate(&mut self, id: u32) {
        if let Some(r) = self.active_refs.get_mut(&id) {
            *r -= 1;
            if *r == 0 {
                self.active_refs.remove(&id);
                if let Some(pos) = self.active_pos.remove(&id) {
                    self.active_vec.swap_remove(pos);
                    if let Some(&moved) = self.active_vec.get(pos) {
                        self.active_pos.insert(moved, pos);
                    }
                }
            }
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            if let Some(scope) = self.scopes.pop() {
                for name in scope.keys() {
                    if let Some(&id) = self.name_ids.get(name) {
                        self.deactivate(id);
                    }
                }
            }
        }
    }

    pub fn current_scope_level(&self) -> usize {
        self.scopes.len() - 1
    }

    /// Creates an entry in the symbol sector (07):
    /// Returns the address token, e.g. "07R0000123" (with condition)
    pub fn define(
        &mut self,
        name: &str,
        target_sector: u8,
        condition: char,
        byte_offset: usize,
    ) -> StructuredToken {
        let slot = self.entries.len() as u32;
        let address_token = StructuredToken::new(Self::SYMBOL_SECTOR, 'R', condition, slot);

        let entry = SymbolEntry {
            name: name.to_string(),
            target_sector,
            slot,
            address_token,
            scope_level: self.current_scope_level(),
            byte_offset,
        };

        // Record in the current scope
        let id = self.intern(name);
        let cur_scope = self.scopes.last_mut().expect("At least one scope must exist");
        if cur_scope.insert(name.to_string(), entry.clone()).is_none() {
            self.activate(id);
        }

        // Record in the global symbol sector entries
        self.entries.push(entry);

        address_token
    }

    /// Resolves an identifier by name looking from innermost to outermost scope.
    pub fn resolve(&self, name: &str) -> Option<&SymbolEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.get(name) {
                return Some(entry);
            }
        }
        None
    }

    /// Ids of all names currently in scope (maintained incrementally).
    pub fn active_ids(&self) -> &[u32] {
        &self.active_vec
    }

    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    pub fn is_active_name(&self, name: &str) -> bool {
        self.name_ids
            .get(name)
            .is_some_and(|id| self.active_pos.contains_key(id))
    }

    /// Names currently in scope (allocates; for tests and diagnostics).
    pub fn active_symbol_names(&self) -> Vec<String> {
        self.active_vec.iter().map(|&i| self.names[i as usize].clone()).collect()
    }
    pub fn get_entry_by_slot(&self, slot: u32) -> Option<&SymbolEntry> {
        self.entries.get(slot as usize)
    }

    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.scopes = vec![HashMap::new()];
        self.entries.clear();
        self.names.clear();
        self.name_ids.clear();
        self.active_vec.clear();
        self.active_pos.clear();
        self.active_refs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_and_resolve() {
        let mut sym_table = SymbolTable::new();
        let addr = sym_table.define("Animal", SectorKind::DefinedName.id(), 'G', 10);

        // Address must be in sector 07 with rule 'R'
        assert_eq!(addr.sector, 7);
        assert_eq!(addr.rule, 'R');
        assert_eq!(addr.id, 0);
        assert_eq!(addr.to_canonical(), "07RG000000");

        let resolved = sym_table.resolve("Animal").expect("Animal should resolve");
        assert_eq!(resolved.name, "Animal");
        assert_eq!(resolved.target_sector, SectorKind::DefinedName.id());
        assert_eq!(resolved.slot, 0);
        assert_eq!(resolved.address_token, addr);

        assert!(sym_table.resolve("NonExistent").is_none());
    }

    #[test]
    fn test_scope_shadowing() {
        let mut sym_table = SymbolTable::new();
        sym_table.define("x", SectorKind::DefinedName.id(), 'G', 0);

        sym_table.push_scope();
        let inner_addr = sym_table.define("x", SectorKind::DefinedName.id(), 'F', 20);
        let resolved = sym_table.resolve("x").unwrap();
        assert_eq!(resolved.address_token, inner_addr);
        assert_eq!(resolved.scope_level, 1);

        sym_table.pop_scope();
        let resolved_outer = sym_table.resolve("x").unwrap();
        assert_eq!(resolved_outer.scope_level, 0);
    }

    #[test]
    fn test_active_ids_follow_scopes() {
        let mut t = SymbolTable::new();
        t.define("a", SectorKind::DefinedName.id(), 'G', 0);
        t.define("x", SectorKind::DefinedName.id(), 'G', 0);
        t.push_scope();
        t.define("x", SectorKind::DefinedName.id(), 'F', 1); // shadows: still one active "x"
        t.define("b", SectorKind::DefinedName.id(), 'F', 2);
        let mut names = t.active_symbol_names();
        names.sort();
        assert_eq!(names, vec!["a", "b", "x"]);
        assert!(t.is_active_name("b"));

        t.pop_scope();
        let mut names = t.active_symbol_names();
        names.sort();
        assert_eq!(names, vec!["a", "x"]);
        assert!(!t.is_active_name("b"));
        assert_eq!(t.active_ids().len(), 2);
    }
}