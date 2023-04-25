use utils::types::Address;
use super::{code::Code, nonce::Nonce};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TxRequest {
    Create {
        nonce: Nonce,
        code: Code,
    },
    Call {
        nonce: Nonce,
        address: Address,
        data: Vec<u8>,
    },
}

impl TxRequest {
    pub fn nonce(&self) -> Nonce {
        match self {
            TxRequest::Call { nonce, .. } | TxRequest::Create { nonce, .. } => *nonce,
        }
    }

    pub fn code(&self) -> Option<Code> {
        match self {
            TxRequest::Create{ code, ..} => {
                return Some(code.clone());
            },
            _ => {
                return None;
            }
        }
    }
}
