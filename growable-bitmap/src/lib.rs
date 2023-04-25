//! A crate providing a growable compact boolean array that can be
//! parameterized on its storage.
//!
//! See the `GrowableBitMap` type for more information.
//!
//! # TODO:
//!
//! This crate is not feature-complete at all. Below are some features I want
//! to add before marking it as `1.0`:
//!
//! - `BitOr` (with another `GrowableBitMap`).
//! - `BitOrAssign` (with another `GrowableBitMap`).
//! - `BitAnd` (with another `GrowableBitMap`).
//! - `BitAndAssign` (with another `GrowableBitMap`).
//! - `BitXor` (with another `GrowableBitMap`).
//! - `BitXorAssign` (with another `GrowableBitMap`).
//!
//! - When `const-generics` become available, possibly use them as storage ?
//!
//! - [Rust 1.48.0+ / Intra-doc links]: Use intra-doc links in documentation.
//!   Right now there are no links because they're painful to write once you've
//!   been introduced to the wonder intra-doc links are.
use std::fmt;
use serde::{Serialize, Deserialize};
mod storage;
pub use storage::BitStorage;

/// A growable compact boolean array that can be parameterized on its storage.
///
/// Bits are stored contiguously. The first value is packed into the least
/// significant bits of the first word of the backing storage.
///
/// The storage must implement the unsafe trait `BitStorage`.
///
/// # Caveats
///
/// - The `GrowableBitMap::set_bit` method may allocate way too much memory
///   compared to what you really need (if for example, you only plan to set
///   the bits between 1200 and 1400). In this case, storing the offset of
///   1200 somewhere else and storing the values in the range `0..=200` in the
///   `GrowableBitMap` is probably the most efficient solution.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct GrowableBitMap<S>
where
    S: BitStorage,
{
    // The storage for the bits.
    bits: Vec<S>,
}

impl<S> fmt::Debug for GrowableBitMap<S>
where
    S: BitStorage + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_list().entries(self.bits.iter()).finish()
    }
}

impl<S> GrowableBitMap<S>
where
    S: BitStorage,
{
    /// Creates a new, empty `GrowableBitMap`.
    ///
    /// This does not allocate anything.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// assert!(GrowableBitMap::<u8>::new().is_empty());
    /// ```
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }

    /// Constructs a new, empty `GrowableBitMap` with the specified capacity
    /// **in bits**.
    ///
    /// When `capacity` is zero, nothing is allocated.
    ///
    /// When `capacity` is not zero, the bit `capacity - 1` can be set without
    /// any other allocation and the returned `GrowableBitMap` is guaranteed
    /// to be able to hold `capacity` bits without reallocating (and maybe more
    /// if the given `capacity` is not a multiple of the number of bits in one
    /// instance of the backing storage).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u16>::with_capacity(16);
    /// assert!(b.is_empty());
    /// assert_eq!(b.capacity(), 16);
    ///
    /// b.set_bit(15);
    /// assert_eq!(b.capacity(), 16);
    ///
    /// b.set_bit(17);
    /// assert!(b.capacity() >= 16);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }

        let div = capacity / S::BITS_IN_STORAGE;
        // Ensures the allocated capacity is enough for values like 125 with a
        // storage of `u8`:
        //
        // - `div` is 15
        // - `capacity % S::BITS_IN_STORAGE` is 5 so `rem` is 1.
        //
        // The final capacity will be 16 `u8`s -> 128 bits, enough for the
        // 125 bits asked for.
        let rem = (capacity % S::BITS_IN_STORAGE != 0) as usize;

        Self {
            bits: Vec::with_capacity(div + rem),
        }
    }

    /// `true` if the `GrowableBitMap` is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u32>::new();
    /// assert!(b.is_empty());
    ///
    /// b.set_bit(25);
    /// assert!(!b.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty() || self.bits.iter().all(|bits| bits.is_empty())
    }

    /// Gets the bit at the given index and returns `true` when it is set to 1,
    /// `false` when it is not.
    ///
    /// This will **not** panic if the index is out of range of the backing
    /// storage, only return `false`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u64>::new();
    /// assert!(!b.get_bit(0));
    /// assert!(!b.get_bit(15));
    ///
    /// b.set_bit(15);
    /// assert!(!b.get_bit(0));
    /// assert!(b.get_bit(15));
    /// ```
    pub fn get_bit(&self, index: usize) -> bool {
        let bits_index = index / S::BITS_IN_STORAGE;

        // Since the bits_index does not exist in the storage, the bit at
        // `index` is logically 0.
        if self.bits.len() <= bits_index {
            return false;
        }

        let elem = &self.bits[bits_index];

        // SAFETY: we have ensure throught the steps above that the index
        // passed to `elem.set_bit` is in range of `0..S::BITS_IN_STORAGE`.
        //
        // Example with a `u8`:
        //
        // `u8::BITS_IN_STORAGE` is 8.
        // `index` is 21.
        //
        // `bits_index` = 2
        // `index - bits_index * S::BITS_IN_STORAGE` = 21 - 2 * 8 = 5 < 8
        unsafe { elem.get_bit(index - bits_index * S::BITS_IN_STORAGE) }
    }

    /// Sets the bit at the given index and returns whether the bit was set
    /// to 1 by this call or not.
    ///
    /// Note: This will grow the backing storage as needed to have enough
    /// storage for the given index. If you set the bit 12800 with a storage of
    /// `u8`s the backing storage will allocate 1600 `u8`s since
    /// `sizeof::<u8>() == 1` byte.
    ///
    /// See also the `Caveats` section on `GrowableBitMap`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u128>::new();
    /// assert!(b.set_bit(0)); // Bit 0 was not set before, returns true.
    /// assert!(!b.set_bit(0)); // Bit 0 was already set, returns false.
    ///
    /// assert!(b.set_bit(255)); // The bitmap will grow as needed to set the bit.
    /// ```
    pub fn set_bit(&mut self, index: usize) -> bool {
        let bits_index = index / S::BITS_IN_STORAGE;

        // Ensure there are enough elements in the `bits` storage.
        if self.bits.len() <= bits_index {
            self.bits.resize_with(bits_index + 1, S::empty);
        }

        let elem = &mut self.bits[bits_index];

        // SAFETY: we have ensure throught the steps above that the index
        // passed to `elem.set_bit` is in range of `0..S::BITS_IN_STORAGE`.
        //
        // Example with a `u8`:
        //
        // `u8::BITS_IN_STORAGE` is 8.
        // `index` is 21.
        //
        // `bits_index` = 2
        // `index - bits_index * S::BITS_IN_STORAGE` = 21 - 2 * 8 = 5 < 8
        unsafe { elem.set_bit(index - bits_index * S::BITS_IN_STORAGE) }
    }

    /// Clears the bit at the given index and returns whether the bit was set
    /// to 0 by this call or not.
    ///
    /// Note: this function will never allocate nor free memory, even when
    /// the bit being cleared is the last 1 in the value at the end of the
    /// backing storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u8>::new();
    /// assert!(!b.clear_bit(3)); // Bit 0 was not set before, returns false.
    ///
    /// b.set_bit(3);
    /// assert!(b.clear_bit(3));
    /// ```
    ///
    /// Testing the effects on capacity:
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u8>::new();
    /// b.set_bit(125);
    ///
    /// let old_capa = b.capacity();
    /// b.clear_bit(125);
    /// assert_eq!(old_capa, b.capacity());
    /// ```
    pub fn clear_bit(&mut self, index: usize) -> bool {
        let bits_index = index / S::BITS_IN_STORAGE;

        // Since the bits_index does not exist in the storage, the bit at
        // `index` is logically 0.
        if self.bits.len() <= bits_index {
            return false;
        }

        let elem = &mut self.bits[bits_index];

        // SAFETY: we have ensure throught the steps above that the index
        // passed to `elem.set_bit` is in range of `0..S::BITS_IN_STORAGE`.
        //
        // Example with a `u8`:
        //
        // `u8::BITS_IN_STORAGE` is 8.
        // `index` is 21.
        //
        // `bits_index` = 2
        // `index - bits_index * S::BITS_IN_STORAGE` = 21 - 2 * 8 = 5 < 8
        unsafe { elem.clear_bit(index - bits_index * S::BITS_IN_STORAGE) }
    }

    /// Clears the bitmap, removing all values (setting them all to 0).
    ///
    /// This method has no effect on the allocated capacity of the bitmap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u16>::new();
    /// b.set_bit(4);
    /// assert!(!b.is_empty());
    ///
    /// b.clear();
    /// assert!(b.is_empty());
    /// ```
    ///
    /// Testing the effects on capacity:
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u16>::new();
    /// b.set_bit(125);
    ///
    /// let old_capa = b.capacity();
    /// b.clear();
    /// assert_eq!(old_capa, b.capacity());
    /// ```
    pub fn clear(&mut self) {
        self.bits.clear();
    }

    /// Counts the number of bits that are set to 1 in the whole bitmap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u32>::new();
    /// assert_eq!(b.count_ones(), 0);
    ///
    /// b.set_bit(2);
    /// assert_eq!(b.count_ones(), 1);
    ///
    /// b.set_bit(9);
    /// assert_eq!(b.count_ones(), 2);
    ///
    /// b.clear();
    /// assert_eq!(b.count_ones(), 0);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bits
            .iter()
            .map(|store| store.count_ones() as usize)
            .sum::<usize>()
    }

    /// Returns the number of bits the bitmap can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u64>::new();
    /// assert_eq!(b.capacity(), 0);
    ///
    /// b.set_bit(380);
    /// assert_eq!(b.capacity(), 384);
    /// ```
    pub fn capacity(&self) -> usize {
        self.bits.capacity() * S::BITS_IN_STORAGE
    }

    /// Shrinks the capacity of the `GrowableBitMap` as much as possible.
    ///
    /// It will drop down as close as possible to the length needed to store
    /// the last bit set to 1 and not more but the allocator may still inform
    /// the bitmap that there is space for a few more elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::{GrowableBitMap, BitStorage};
    ///
    /// let mut b = GrowableBitMap::<u128>::with_capacity(381);
    ///
    /// b.set_bit(127);
    /// b.set_bit(380);
    /// assert_eq!(b.capacity(), 384);
    ///
    /// b.clear_bit(380);
    /// b.shrink_to_fit();
    /// assert_eq!(b.capacity(), 128);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        // Ignoring the values at the end that are 0.
        let last_set_bit_index = self
            .bits
            .iter()
            .rev()
            .skip_while(|&store| store.is_empty())
            .count();

        self.bits.truncate(last_set_bit_index);
        self.bits.shrink_to_fit();
    }
}

macro_rules! growable_bitmap_storage_integer_tests {
    ($(($int: ty, $mod_name: ident))*) => {
        $(
            #[cfg(test)]
            mod $mod_name {
                use super::*;

                #[test]
                fn new() {
                    let bm = GrowableBitMap::<$int>::new();
                    let bm2 = GrowableBitMap {
                        bits: Vec::<$int>::new(),
                    };

                    assert_eq!(bm, bm2);
                }

                #[test]
                fn with_capacity() {
                    let bm = GrowableBitMap::<$int>::with_capacity(0);
                    let vec = Vec::<$int>::with_capacity(0);
                    assert_eq!(bm.bits.capacity(), vec.capacity());

                    let bm = GrowableBitMap::<$int>::with_capacity(11);
                    let vec = Vec::<$int>::with_capacity(11);
                    assert!(bm.bits.capacity() >= vec.capacity() / <$int>::BITS_IN_STORAGE);
                }

                #[test]
                fn is_empty() {
                    let bm = GrowableBitMap::<$int>::new();
                    assert!(bm.is_empty());

                    let mut bm = GrowableBitMap::<$int>::with_capacity(3);
                    assert!(bm.is_empty());

                    bm.set_bit(7);
                    assert!(!bm.is_empty());

                    bm.clear_bit(7);
                    assert!(bm.is_empty());

                    bm.set_bit(7);
                    bm.set_bit(3);
                    bm.set_bit(4);
                    assert!(!bm.is_empty());

                    bm.clear();
                    assert!(bm.is_empty());
                }

                #[test]
                fn get_bit() {
                    let bm = GrowableBitMap::<$int>::new();

                    assert!(!bm.get_bit(0));
                    assert!(!bm.get_bit(7));

                    // Ensuring `get_bit` does not allocate.
                    let old_capa = bm.capacity();
                    assert!(!bm.get_bit(200000003));
                    assert_eq!(old_capa, bm.capacity());

                    let mut bm = bm;
                    bm.set_bit(0);
                    bm.set_bit(6);

                    assert!(bm.get_bit(0));
                    assert!(bm.get_bit(6));

                    // Still false.
                    assert!(!bm.get_bit(7));
                    assert!(!bm.get_bit(3));
                }

                #[test]
                fn set_bit() {
                    let mut bm = GrowableBitMap::<$int>::new();

                    assert!(bm.set_bit(0));
                    assert!(bm.set_bit(7));

                    assert!(!bm.set_bit(0));

                    // Ensuring `set_bit` does allocate when necessary.
                    let old_capa = bm.capacity();
                    assert!(bm.set_bit(150));
                    assert!(old_capa <= bm.capacity());
                    assert!(bm.get_bit(150));
                }

                #[test]
                fn clear_bit() {
                    let mut bm = GrowableBitMap::<$int>::new();

                    assert!(!bm.clear_bit(0));
                    assert!(!bm.clear_bit(7));

                    bm.set_bit(0);
                    assert!(bm.clear_bit(0));

                    // Ensuring `clear_bit` does not allocate nor free.
                    let old_capa = bm.capacity();
                    assert!(!bm.clear_bit(200000003));
                    assert_eq!(old_capa, bm.capacity());

                    bm.set_bit(150);
                    assert!(bm.clear_bit(150));

                    assert!(bm.is_empty());
                }

                #[test]
                fn clear() {
                    let mut bm = GrowableBitMap::<$int>::new();

                    bm.clear();
                    assert!(bm.is_empty());

                    bm.set_bit(253);
                    // Ensuring `clear_bit` does not allocate nor free.
                    let old_capa = bm.capacity();
                    bm.clear();
                    assert_eq!(old_capa, bm.capacity());

                    bm.set_bit(30);
                    bm.set_bit(4);

                    bm.clear();
                    assert!(bm.is_empty());
               }

                #[test]
                fn count_ones() {
                    let mut bm = GrowableBitMap::<$int>::new();
                    assert_eq!(bm.count_ones(), 0);

                    bm.set_bit(0);
                    assert_eq!(bm.count_ones(), 1);

                    bm.set_bit(30);
                    assert_eq!(bm.count_ones(), 2);

                    bm.set_bit(7);
                    assert_eq!(bm.count_ones(), 3);

                    bm.clear_bit(0);
                    assert_eq!(bm.count_ones(), 2);

                    bm.clear();
                    assert_eq!(bm.count_ones(), 0);
                }

                #[test]
                fn capacity() {
                    let mut bm = GrowableBitMap::<$int>::new();
                    assert_eq!(bm.capacity(), 0);

                    bm.set_bit(511);
                    assert_eq!(bm.capacity(), 512);
                }

                #[test]
                fn shrink_to_fit() {
                    let mut bm = GrowableBitMap::<$int>::with_capacity(381);
                    bm.set_bit(127);
                    bm.set_bit(380);
                    assert_eq!(bm.capacity(), 384);

                    bm.clear_bit(380);
                    bm.shrink_to_fit();
                    assert_eq!(bm.capacity(), 128);
                }
            }
         )*
    }
}

growable_bitmap_storage_integer_tests! {
    (u8, test_u8)
    (u16, test_u16)
    (u32, test_u32)
    (u64, test_u64)
    (u128, test_u128)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_serialize() {
        let mut bitmap = GrowableBitMap::<u8>::new();
        bitmap.set_bit(1);
        bitmap.set_bit(2);
        let v = bincode::serialize(&bitmap).unwrap();
        let load: GrowableBitMap<u8> = bincode::deserialize(&v).unwrap();
        assert_eq!(load, bitmap);
    }
}