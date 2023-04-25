use serde::{Serialize, Deserialize};
use super::{H160, H256, Params};
use blake2b_simd::Hash;
use std::fmt::Debug;
use rug::{Integer, integer::Order};
use std::convert::{From, Into};
use primitive_types::{H160 as ETHH160, H256 as ETHH256};
pub const ACC_ADDR_SIZE: usize = 20;
pub const STATE_ADDR_SIZE: usize = 32;
pub const VERSION_SIZE: usize = 4;
pub const VALUE_SIZE: usize = 32;
pub const COMPOUND_KEY_SIZE: usize = ACC_ADDR_SIZE + STATE_ADDR_SIZE + VERSION_SIZE;
pub const STATE_SIZE: usize = COMPOUND_KEY_SIZE + VALUE_SIZE;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, derive_more::Deref, derive_more::DerefMut, derive_more::Display, derive_more::From, derive_more::Into)]
pub struct Address(pub H160);

impl From<ETHH160> for Address {
    fn from(value: ETHH160) -> Self {
        Address(H160::from(value.0))
    }
}

/*
key of account address (160-bits) and state address (256-bit)
 */
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AddrKey {
    pub acc_addr: Address,
    pub state_addr: StateKey,
}

impl AddrKey {
    pub fn new(acc_addr: Address, state_addr: StateKey) -> Self {
        Self {
            acc_addr,
            state_addr,
        }
    }

    /* Serialize manually to a byte string
     */
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.acc_addr.as_bytes().to_vec();
        bytes.extend(self.state_addr.as_bytes());
        return bytes;
    }
}
/*
Compound Key consists of the address (including the account and state addresses) and the version that the state is updated (u32)
 */
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompoundKey {
    pub addr: AddrKey,
    pub version: u32,
}

impl CompoundKey {
    pub fn new(acc_addr: Address, state_addr: StateKey, version: u32) -> Self {
        let addr = AddrKey::new(acc_addr, state_addr);
        Self {
            addr,
            version,
        }
    }

    pub fn new_with_addr_key(addr_key: AddrKey, version: u32) -> Self {
        Self {
            addr: addr_key,
            version,
        }
    }

    /* Serialize manually to a byte string
     */
    pub fn to_bytes(&self) -> Vec<u8> {
        let addr = self.addr;
        let acc_addr = addr.acc_addr;
        let state_addr = addr.state_addr;
        let version = self.version;
        let mut bytes = acc_addr.as_bytes().to_vec();
        bytes.extend(state_addr.as_bytes());
        bytes.extend(&version.to_be_bytes());
        return bytes;
    }
    /* Deserialize manually from a byte string to a compound key
     */
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), ACC_ADDR_SIZE + STATE_ADDR_SIZE + VERSION_SIZE);
        let acc_addr = H160::from_slice(&bytes[0..ACC_ADDR_SIZE]);
        let state_addr = H256::from_slice(&bytes[ACC_ADDR_SIZE..ACC_ADDR_SIZE+STATE_ADDR_SIZE]);
        let version = u32::from_be_bytes(bytes[ACC_ADDR_SIZE+STATE_ADDR_SIZE..COMPOUND_KEY_SIZE].try_into().expect("error"));
        let addr_key = AddrKey {
            acc_addr: acc_addr.into(),
            state_addr: state_addr.into(),
        };
        Self {
            addr: addr_key,
            version,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, derive_more::Deref, derive_more::DerefMut, derive_more::Display, derive_more::From, derive_more::Into)]
pub struct StateKey(pub H256);

impl From<ETHH256> for StateKey {
    fn from(value: ETHH256) -> Self {
        StateKey(H256::from(value.0))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, derive_more::Deref, derive_more::DerefMut, derive_more::Display, derive_more::From, derive_more::Into)]
pub struct StateValue(pub H256);

impl From<ETHH256> for StateValue {
    fn from(value: ETHH256) -> Self {
        StateValue(H256::from(value.0))
    }
}

impl Into<ETHH256> for StateValue {
    fn into(self) -> ETHH256 {
        let value = self.0;
        ETHH256::from(value.0)
    }
}

impl StateValue {
    /* Serialize manually to a byte string
     */
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = self.as_bytes().to_vec();
        return bytes;
    }
    /* Deserialize manually from a byte string to a compound key-value pair
     */
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 32);
        let value = H256::from_slice(&bytes);
        Self(value)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompoundKeyValue {
    pub key: CompoundKey,
    pub value: StateValue,
}

impl CompoundKeyValue {
    pub fn new(acc_addr: Address, state_addr: StateKey, version: u32, value: StateValue) -> Self {
        let key = CompoundKey::new(acc_addr, state_addr, version);
        Self {
            key,
            value,
        }
    }

    pub fn new_with_compound_key(compound_key: CompoundKey, value: StateValue) -> Self {
        Self {
            key: compound_key,
            value,
        }
    }
    
    /* Serialize manually to a byte string
     */
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.key.to_bytes();
        bytes.extend(self.value.as_bytes());
        return bytes;
    }
    /* Deserialize manually from a byte string to a compound key-value pair
     */
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), STATE_SIZE);
        let key = CompoundKey::from_bytes(&bytes[0..COMPOUND_KEY_SIZE]);
        let value = H256::from_slice(&bytes[COMPOUND_KEY_SIZE..STATE_SIZE]);
        Self {
            key,
            value: value.into(),
        }
    }
}

pub trait Digestible {
    fn to_digest(&self) -> H256;
}

impl Digestible for AddrKey {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

impl Digestible for CompoundKeyValue {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

impl Digestible for CompoundKey {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

impl Digestible for StateValue {
    fn to_digest(&self) -> H256 {
        let bytes = self.to_bytes();
        bytes_hash(&bytes)
    }
}

pub fn compute_concatenate_hash(v: &[H256]) -> H256 {
    let mut bytes = vec![];
    for elem in v {
        bytes.extend(elem.as_bytes());
    }
    bytes_hash(&bytes)
}

pub fn bytes_hash(bytes: &[u8]) -> H256 {
    let mut hasher = Params::new().hash_length(32).to_state();
    hasher.update(bytes);
    let h = H256::from_slice(hasher.finalize().as_bytes());
    return h;
}

#[inline]
pub fn blake2b_hash_to_h160(input: Hash) -> H160 {
    H160::from_slice(input.as_bytes())
}

#[inline]
pub fn blake2b_hash_to_h256(input: Hash) -> H256 {
    H256::from_slice(input.as_bytes())
}

pub const DEFAULT_DIGEST_LEN: usize = 32;

#[inline]
pub fn blake2(size: usize) -> Params {
    let mut params = Params::new();
    params.hash_length(size);
    params
}

#[inline]
pub fn default_blake2() -> Params {
    blake2(DEFAULT_DIGEST_LEN)
}

impl Digestible for [u8] {
    fn to_digest(&self) -> H256 {
        let hash = default_blake2().hash(self);
        blake2b_hash_to_h256(hash)
    }
}

impl Digestible for std::vec::Vec<u8> {
    fn to_digest(&self) -> H256 {
        self.as_slice().to_digest()
    }
}

impl Digestible for str {
    fn to_digest(&self) -> H256 {
        self.as_bytes().to_digest()
    }
}

impl Digestible for std::string::String {
    fn to_digest(&self) -> H256 {
        self.as_bytes().to_digest()
    }
}

macro_rules! impl_digestible_for_numeric {
    ($x: ty) => {
        impl Digestible for $x {
            fn to_digest(&self) -> H256 {
                self.to_le_bytes().to_digest()
            }
        }
    };
    ($($x: ty),*) => {$(impl_digestible_for_numeric!($x);)*}
}

impl_digestible_for_numeric!(i8, i16, i32, i64);
impl_digestible_for_numeric!(u8, u16, u32, u64);
impl_digestible_for_numeric!(f32, f64);
pub trait Num:
    PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Clone
    + Copy
    + Digestible
    + Default
    + Debug
    + Sized
    + Serialize
{

}

impl<T> Num for T where
    T: PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Clone
    + Copy
    + Digestible
    + Default
    + Debug
    + Serialize
{

}

pub trait Value:
    PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Digestible
    + Clone
    + Copy
    + Default
    + Debug
    + Serialize
{

}

impl<T> Value for T where
    T: PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Digestible
    + Clone
    + Copy
    + Default
    + Debug
    + Serialize
{

}

/* A trait to transform a type to the big integer type
 */
pub trait BigNum {
    fn to_big_integer(&self,) -> Integer;
}

impl BigNum for AddrKey {
    fn to_big_integer(&self,) -> Integer {
        let bytes = self.to_bytes();
        Integer::from_digits(&bytes, Order::Msf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /*
    Test Serialize/Deserialize Compound Key-Value pair
     */
    #[test]
    fn test_compound_key_value() {
        let compound_key_value = CompoundKeyValue::new(Address(H160::random()), StateKey(H256::random()), 10, StateValue(H256::random()));
        println!("hash: {:?}", compound_key_value.to_digest());
        let bytes = compound_key_value.to_bytes();
        let decode = CompoundKeyValue::from_bytes(&bytes);
        assert_eq!(decode, compound_key_value);
    }

    #[test]
    fn test_bytes() {
        let compound_key_value = CompoundKeyValue::new(Address(H160::random()), StateKey(H256::random()), 10, StateValue(H256::random()));
        let bytes = bincode::serialize(&compound_key_value).unwrap();
        println!("len: {}", bytes.len());
        let read: CompoundKeyValue = bincode::deserialize(&bytes).unwrap();
        assert_eq!(read, compound_key_value);
    }
}
