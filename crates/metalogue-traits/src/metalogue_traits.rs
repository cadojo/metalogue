// This is not a module! This source code is included
// directly in `lib.rs`.

pub trait Device {
    fn acquire() -> Self;
}
