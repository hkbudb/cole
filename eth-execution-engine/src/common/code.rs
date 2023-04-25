use core::fmt;
#[derive(
    Default,
    Clone,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::From,
    derive_more::Into,
)]
pub struct Code(pub Vec<u8>);

impl fmt::Debug for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Code [len={}]", self.0.len())
    }
}

impl Code {
    pub fn new() -> Self {
        Self::default()
    }
}