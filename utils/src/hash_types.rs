use serde::{Serialize, Deserialize};

fixed_hash::construct_fixed_hash!{
    /// 256 bit hash type
    pub struct H256(32);
}

impl Serialize for H256 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer {        
        self.as_fixed_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for H256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de> {
        let v = <[u8; 32]>::deserialize(deserializer)?;
        Ok(H256::from(v))
    }
}

fixed_hash::construct_fixed_hash!{
    /// 160 bit hash type
    pub struct H160(20);
}

impl Serialize for H160 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer {        
        self.as_fixed_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for H160 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de> {
        let v = <[u8; 20]>::deserialize(deserializer)?;
        Ok(H160::from(v))
    }
}