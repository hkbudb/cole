use utils::types::{Address, StateKey, StateValue};
use super::{code::Code, nonce::Nonce};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, HashSet, hash_map, btree_map};

bitflags! {
    #[derive(Default, Serialize, Deserialize)]
    pub struct ReadAccessFlags: u8 {
        const NONCE = 0b001;
        const CODE  = 0b010;
    }
}

impl ReadAccessFlags {
    pub fn get_nonce(self) -> bool {
        self.contains(Self::NONCE)
    }

    pub fn get_code(self) -> bool {
        self.contains(Self::CODE)
    }

    pub fn set_nonce(&mut self, value: bool) {
        self.set(Self::NONCE, value);
    }

    pub fn set_code(&mut self, value: bool) {
        self.set(Self::CODE, value);
    }
}

bitflags! {
    #[derive(Default, Serialize, Deserialize)]
    pub struct WriteAccessFlags: u8 {
        const NONCE        = 0b001;
        const CODE         = 0b010;
        const RESET_VALUES = 0b100;
    }
}

impl WriteAccessFlags {
    pub fn get_nonce(self) -> bool {
        self.contains(Self::NONCE)
    }

    pub fn get_code(self) -> bool {
        self.contains(Self::CODE)
    }

    pub fn get_reset_values(self) -> bool {
        self.contains(Self::RESET_VALUES)
    }

    pub fn set_nonce(&mut self, value: bool) {
        self.set(Self::NONCE, value);
    }

    pub fn set_code(&mut self, value: bool) {
        self.set(Self::CODE, value);
    }

    pub fn set_reset_values(&mut self, value: bool) {
        self.set(Self::RESET_VALUES, value);
    }
}

#[derive(
    Debug,
    Default,
    Clone,
    Eq,
    PartialEq,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
)]
pub struct TxReadSet(pub HashMap<Address, AccountReadSet>);

#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct AccountReadSet {
    pub access_flags: ReadAccessFlags,
    pub values: HashSet<StateKey>,
}

impl AccountReadSet {
    pub fn get_nonce(&self) -> bool {
        self.access_flags.get_nonce()
    }

    pub fn set_nonce(&mut self, value: bool) {
        self.access_flags.set_nonce(value);
    }

    pub fn get_code(&self) -> bool {
        self.access_flags.get_code()
    }

    pub fn set_code(&mut self, value: bool) {
        self.access_flags.set_code(value);
    }

    pub fn get_values(&self) -> &HashSet<StateKey> {
        &self.values
    }

    pub fn value_iter(&self) -> impl Iterator<Item = &'_ StateKey> {
        self.values.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.access_flags.is_empty() && self.values.is_empty()
    }
}

#[derive(
    Debug,
    Default,
    Clone,
    Eq,
    PartialEq,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
)]
pub struct TxReadData(pub HashMap<Address, AccountReadData>);

impl TxReadData {
    pub fn to_set(&self) -> TxReadSet {
        let mut out = TxReadSet::default();
        for (k, v) in &self.0 {
            out.0.insert(*k, v.to_set());
        }
        out
    }

    pub fn get_nonce(&self, address: Address) -> Option<Nonce> {
        self.0.get(&address).and_then(|acc| acc.nonce)
    }

    pub fn add_nonce(&mut self, address: Address, nonce: Nonce) {
        self.0.entry(address).or_default().nonce = Some(nonce);
    }

    pub fn remove_nonce(&mut self, address: Address) {
        match self.0.entry(address) {
            hash_map::Entry::Occupied(mut e) => {
                let acc_data = e.get_mut();
                acc_data.nonce = None;
                if acc_data.is_empty() {
                    e.remove();
                }
            }
            hash_map::Entry::Vacant(_) => {}
        }
    }

    pub fn get_or_add_nonce(&mut self, address: Address, f: impl FnOnce() -> Nonce) -> Nonce {
        let acc_data = self.0.entry(address).or_default();
        *acc_data.nonce.get_or_insert_with(f)
    }

    pub fn get_code(&self, address: Address) -> Option<&Code> {
        self.0.get(&address).and_then(|acc| acc.code.as_ref())
    }

    pub fn add_code(&mut self, address: Address, code: Code) {
        self.0.entry(address).or_default().code = Some(code);
    }

    pub fn get_or_add_code(&mut self, address: Address, f: impl FnOnce() -> Code) -> &Code {
        let acc_data = self.0.entry(address).or_default();
        acc_data.code.get_or_insert_with(f)
    }

    pub fn get_value(&self, address: Address, key: StateKey) -> Option<StateValue> {
        self.0
            .get(&address)
            .and_then(|acc| acc.values.get(&key).copied())
    }

    pub fn add_value(&mut self, address: Address, key: StateKey, value: StateValue) {
        let acc_data = self.0.entry(address).or_default();
        *acc_data.values.entry(key).or_default() = value;
    }

    pub fn get_or_add_value(
        &mut self,
        address: Address,
        key: StateKey,
        f: impl FnOnce() -> StateValue,
    ) -> StateValue {
        let acc_data = self.0.entry(address).or_default();
        *acc_data.values.entry(key).or_insert_with(f)
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct AccountReadData {
    pub nonce: Option<Nonce>,
    pub code: Option<Code>,
    pub values: HashMap<StateKey, StateValue>,
}

impl AccountReadData {
    pub fn to_set(&self) -> AccountReadSet {
        let mut access_flags = ReadAccessFlags::empty();
        access_flags.set_nonce(self.nonce.is_some());
        access_flags.set_code(self.code.is_some());
        AccountReadSet {
            access_flags,
            values: self.values.keys().copied().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nonce.is_none() && self.code.is_none() && self.values.is_empty()
    }
}

#[derive(
    Debug,
    Default,
    Clone,
    Eq,
    PartialEq,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
)]
pub struct TxWriteData(pub BTreeMap<Address, AccountWriteData>);

impl TxWriteData {
    pub fn delete_account(&mut self, address: Address) {
        self.0.insert(
            address,
            AccountWriteData {
                nonce: Some(Nonce::default()),
                code: Some(Code::new()),
                values: BTreeMap::new(),
                reset_values: true,
            },
        );
    }

    pub fn add_nonce(&mut self, address: Address, nonce: Nonce) {
        self.0.entry(address).or_default().nonce = Some(nonce);
    }

    pub fn add_code(&mut self, address: Address, code: Code) {
        self.0.entry(address).or_default().code = Some(code);
    }

    pub fn add_reset_values(&mut self, address: Address) {
        self.0.entry(address).or_default().reset_values = true;
    }

    pub fn add_value(&mut self, address: Address, key: StateKey, value: StateValue) {
        *self
            .0
            .entry(address)
            .or_default()
            .values
            .entry(key)
            .or_default() = value;
    }

    pub fn merge(&mut self, new: &TxWriteData) {
        for (&k, v) in new.0.iter() {
            match self.entry(k) {
                btree_map::Entry::Occupied(mut e) => {
                    e.get_mut().merge(v);
                }
                btree_map::Entry::Vacant(e) => {
                    e.insert(v.clone());
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct AccountWriteData {
    pub nonce: Option<Nonce>,
    pub code: Option<Code>,
    pub values: BTreeMap<StateKey, StateValue>,
    pub reset_values: bool,
}

impl AccountWriteData {
    pub fn merge(&mut self, new: &AccountWriteData) {
        if new.nonce.is_some() {
            self.nonce = new.nonce;
        }

        if new.code.is_some() {
            self.code = new.code.clone();
        }

        if new.reset_values {
            self.reset_values = true;
            self.values = new.values.clone();
        } else {
            self.values.extend(new.values.iter());
        }
    }

    pub fn has_nonce(&self) -> bool {
        self.nonce.is_some()
    }

    pub fn has_code(&self) -> bool {
        self.code.is_some()
    }

    pub fn has_reset_values(&self) -> bool {
        self.reset_values
    }

    pub fn value_keys(&self) -> impl Iterator<Item = &'_ StateKey> {
        self.values.keys()
    }
}