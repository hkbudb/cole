/// Named to clarify bit/byte operations.
pub const BITS_IN_BYTE: usize = 8;

/// Types implementing this trait can be used as storage for a `GrowableBitmap`.
///
/// Only fixed-size types should implement this: see the `BITS_IN_STORAGE`
/// constant requirement for this trait.
///
/// # Safety
///
/// This trait exposes several methods that are `unsafe` to call.
///
/// The given `index` must fit in `0..<Self as BitStorage>::BITS_IN_STORAGE`
/// for the behaviour of the `unsafe` methods to be correct.
pub unsafe trait BitStorage: Sized {
    /// Number of bits that can be stored in one instance of `Self`.
    ///
    /// This is a constant and types implementing this trait guarantee this
    /// will always be exact.
    const BITS_IN_STORAGE: usize = std::mem::size_of::<Self>() * BITS_IN_BYTE;

    /// Construct a new, empty instance of a `BitStorage` type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let a = u8::empty();
    /// assert_eq!(a, 0);
    /// ```
    fn empty() -> Self;

    /// Returns `true` is the storage is considered empty (no `1`s anywhere).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let a = u16::empty();
    /// assert!(a.is_empty());
    ///
    /// let a = 1_u16 << 2;
    /// assert!(!a.is_empty());
    /// ```
    fn is_empty(&self) -> bool;

    /// Gets the bit at the given index and returns `true` when it is set
    /// to 1, `false` when it is not.
    ///
    /// # Safety
    ///
    /// The given `index` must fit in `0..<Self as BitStorage>::BITS_IN_STORAGE`
    /// for the behaviour of this method to be correct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let a = u32::empty();
    /// assert!(unsafe { !a.get_bit(2) });
    ///
    /// let a = 1_u32 << 2;
    /// assert!(unsafe { a.get_bit(2) });
    /// ```
    unsafe fn get_bit(&self, index: usize) -> bool;

    /// Sets the bit at the given index and returns `true` when it is set
    /// to 1 by this call, `false` when it was already 1.
    ///
    /// # Safety
    ///
    /// The given `index` must fit in `0..<Self as BitStorage>::BITS_IN_STORAGE`
    /// for the behaviour of this method to be correct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let mut a = u64::empty();
    ///
    /// assert!(unsafe { a.set_bit(0) });
    /// assert!(unsafe { a.get_bit(0) });
    ///
    /// assert!(unsafe { a.set_bit(7) });
    /// assert!(unsafe { a.get_bit(7) });
    ///
    /// assert!(unsafe { !a.set_bit(0) });
    /// ```
    unsafe fn set_bit(&mut self, index: usize) -> bool;

    /// Clears the bit at the given index and returns `true` when it is set
    /// to 0 by this call, `false` when it was already 0.
    ///
    /// # Safety
    ///
    /// The given `index` must fit in `0..<Self as BitStorage>::BITS_IN_STORAGE`
    /// for the behaviour of this method to be correct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let mut a = u64::empty();
    ///
    /// assert!(unsafe { !a.clear_bit(56) });
    ///
    /// assert!(unsafe { a.set_bit(56) });
    /// assert!(unsafe { a.clear_bit(56) });
    /// assert!(unsafe { !a.get_bit(56) });
    /// ```
    unsafe fn clear_bit(&mut self, index: usize) -> bool;

    /// Clears the whole storage, setting `self` to the empty value.
    ///
    /// The default implementation uses `*self = Self::empty()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let mut a = u128::empty();
    ///
    /// let mut a: u128 = 42;
    /// a.clear_all();
    /// assert!(a.is_empty());
    ///
    /// let mut a: u128 = 1 << 120;
    /// a.clear_all();
    /// assert!(a.is_empty());
    /// ```
    fn clear_all(&mut self) {
        *self = Self::empty();
    }

    /// Returns the number of bits set to 1 in `Self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let a = u8::empty();
    /// assert_eq!(a.count_ones(), 0);
    ///
    /// let mut a = a;
    /// unsafe { a.set_bit(0); }
    /// assert_eq!(a.count_ones(), 1);
    ///
    /// unsafe { a.set_bit(3); }
    /// assert_eq!(a.count_ones(), 2);
    ///
    /// unsafe { a.set_bit(7); }
    /// assert_eq!(a.count_ones(), 3);
    ///
    /// unsafe { a.clear_bit(4); }
    /// assert_eq!(a.count_ones(), 3);
    ///
    /// unsafe { a.clear_bit(7); }
    /// assert_eq!(a.count_ones(), 2);
    ///
    /// a.clear_all();
    /// assert_eq!(a.count_ones(), 0);
    /// ```
    fn count_ones(&self) -> usize;

    /// Returns the index of the first bit set in the binary representation of
    /// `self`, if any, else returns `None`.
    ///
    /// # Example
    ///
    /// Example on a u8 where the least significant bit is to the right:
    ///
    /// ```text
    /// 0b1100_0100
    ///         ^
    ///         index = 3
    /// ```
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let v = u16::empty();
    /// assert_eq!(v.first_bit_set(), None);
    ///
    /// let mut v = v;
    /// unsafe { v.set_bit(3); }
    /// assert_eq!(v.first_bit_set(), Some(3));
    ///
    /// unsafe { v.set_bit(7); }
    /// assert_eq!(v.first_bit_set(), Some(3));
    ///
    /// unsafe { v.set_bit(1); }
    /// assert_eq!(v.first_bit_set(), Some(1));
    /// ```
    fn first_bit_set(&self) -> Option<usize>;

    /// Returns the index of the last bit set in the binary representation of
    /// `self`, if any, else returns `None`.
    ///
    /// # Example
    ///
    /// Example on a u8 where the least significant bit is to the right:
    ///
    /// ```text
    /// 0b0010_0011
    ///     ^
    ///     index = 5
    /// ```
    ///
    /// ```rust
    /// use growable_bitmap::BitStorage;
    ///
    /// let v = u16::empty();
    /// assert_eq!(v.last_bit_set(), None);
    ///
    /// let mut v = v;
    /// unsafe { v.set_bit(3); }
    /// assert_eq!(v.last_bit_set(), Some(3));
    ///
    /// unsafe { v.set_bit(1); }
    /// assert_eq!(v.last_bit_set(), Some(3));
    ///
    /// unsafe { v.set_bit(7); }
    /// assert_eq!(v.last_bit_set(), Some(7));
    /// ```
    fn last_bit_set(&self) -> Option<usize>;
}

macro_rules! bit_storage_integer_impl {
    ($int: ty, $doc: expr) => {
        #[doc = $doc]
        unsafe impl BitStorage for $int {
            #[inline(always)]
            fn empty() -> Self { 0 }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                *self == Self::empty()
            }

            unsafe fn get_bit(&self, index: usize) -> bool {
                let mask = 1 << index;
                (*self & mask) != 0
            }

            unsafe fn set_bit(&mut self, index: usize) -> bool {
                let mask = 1 << index;
                let prev = *self & mask;

                *self |= mask;
                prev == 0
            }

            unsafe fn clear_bit(&mut self, index: usize) -> bool {
                let mask = 1 << index;
                let prev = *self & mask;

                *self &= !mask;
                prev != 0
            }

            #[inline(always)]
            fn count_ones(&self) -> usize {
                <$int>::count_ones(*self) as usize
            }

            fn first_bit_set(&self) -> Option<usize> {
                if *self == Self::empty() {
                    None
                } else {
                    // Example on a u8 where the least significant bit
                    // is to the right:
                    //
                    // 0b1100_0100
                    //         ^
                    //         index = 3
                    let mut mask = 1;
                    for idx in 0..Self::BITS_IN_STORAGE {
                        if *self & mask != 0 { return Some(idx); }
                        mask <<= 1;
                    }
                    // Since the case for 0 has been checked for earlier,
                    // it is guaranteed that at least one bit is set to 1.
                    unreachable!()
                }
            }

            fn last_bit_set(&self) -> Option<usize> {
                if *self == Self::empty() {
                    None
                } else {
                    // Example on a u8 where the least significant bit
                    // is to the right:
                    //
                    // 0b0010_0011
                    //     ^
                    //     index = 5
                    let mut mask = 1 << (Self::BITS_IN_STORAGE - 1);
                    for idx in (0..=(Self::BITS_IN_STORAGE - 1)).rev() {
                        if *self & mask != 0 { return Some(idx); }
                        mask >>= 1;
                    }
                    // Since the case for 0 has been checked for earlier,
                    // it is guaranteed that at least one bit is set to 1.
                    unreachable!()
                }
            }
        }
    };
    ($int: ty) => {
        bit_storage_integer_impl! {
            $int,
            concat!(
                "SAFETY: this implementation is safe because the width in ",
                "bits of an `",
                stringify!($int),
                "` is fixed and equal to `std::mem::size_of::<",
                stringify!($int),
                ">() * 8`.",
            )
        }
    };
}

bit_storage_integer_impl! { u8 }
bit_storage_integer_impl! { u16 }
bit_storage_integer_impl! { u32 }
bit_storage_integer_impl! { u64 }
bit_storage_integer_impl! { u128 }

macro_rules! bit_storage_integer_tests {
    ($(($int: ty, $mod_name: ident))*) => {
        $(
            #[cfg(test)]
            mod $mod_name {
                use super::*;

                #[test]
                fn empty() {
                    assert_eq!(<$int>::empty(), 0);
                }

                #[test]
                fn is_empty() {
                    assert!(<$int>::empty().is_empty());
                    let v: $int = 3;
                    assert!(!v.is_empty());
                }

                #[test]
                fn get_bit() {
                    let v = <$int>::empty();
                    assert!(unsafe { !v.get_bit(0) });
                    assert!(unsafe { !v.get_bit(7) });

                    let v: $int = 1 << 3;
                    assert!(unsafe { !v.get_bit(0) });
                    assert!(unsafe { v.get_bit(3) });
                }

                #[test]
                fn set_bit() {
                    let mut v = <$int>::empty();

                    assert!(unsafe { v.set_bit(0) });
                    assert!(unsafe { v.get_bit(0) });

                    assert!(unsafe { v.set_bit(7) });
                    assert!(unsafe { v.get_bit(7) });

                    assert!(unsafe { !v.set_bit(0) });
                }

                #[test]
                fn clear_bit() {
                    let mut v = <$int>::empty();

                    assert!(unsafe { !v.clear_bit(0) });

                    assert!(unsafe { v.set_bit(0) });
                    assert!(unsafe { v.clear_bit(0) });
                    assert!(unsafe { !v.get_bit(0) });
                }

                #[test]
                fn clear_all() {
                    let mut v: $int = 42;
                    v.clear_all();
                    assert!(v.is_empty());

                    let mut v: $int = 1 << 3;
                    v.clear_all();
                    assert!(v.is_empty());
                }

                #[test]
                fn count_ones() {
                    let v = <$int>::empty();
                    assert_eq!(v.count_ones(), 0);

                    let mut v = v;
                    unsafe { v.set_bit(0); }
                    assert_eq!(v.count_ones(), 1);

                    unsafe { v.set_bit(3); }
                    assert_eq!(v.count_ones(), 2);

                    unsafe { v.set_bit(7); }
                    assert_eq!(v.count_ones(), 3);

                    unsafe { v.clear_bit(4); }
                    assert_eq!(v.count_ones(), 3);

                    unsafe { v.clear_bit(7); }
                    assert_eq!(v.count_ones(), 2);

                    v.clear_all();
                    assert_eq!(v.count_ones(), 0);
                }

                #[test]
                fn first_bit_set() {
                    let v = <$int>::empty();
                    assert_eq!(v.first_bit_set(), None);

                    let mut v = v;
                    unsafe { v.set_bit(3); }
                    assert_eq!(v.first_bit_set(), Some(3));

                    unsafe { v.set_bit(7); }
                    assert_eq!(v.first_bit_set(), Some(3));

                    unsafe { v.set_bit(1); }
                    assert_eq!(v.first_bit_set(), Some(1));
                }

                #[test]
                fn last_bit_set() {
                    let v = <$int>::empty();
                    assert_eq!(v.last_bit_set(), None);

                    let mut v = v;
                    unsafe { v.set_bit(3); }
                    assert_eq!(v.last_bit_set(), Some(3));

                    unsafe { v.set_bit(1); }
                    assert_eq!(v.last_bit_set(), Some(3));

                    unsafe { v.set_bit(7); }
                    assert_eq!(v.last_bit_set(), Some(7));
                }
            }
        )*
    }
}

bit_storage_integer_tests! {
    (u8, test_u8)
    (u16, test_u16)
    (u32, test_u32)
    (u64, test_u64)
    (u128, test_u128)
}