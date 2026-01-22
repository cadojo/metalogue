// This is not a module! This source code is included
// directly in `lib.rs`.

use mlx_rs;

/// A thin wrapper around `mlx_rs::Device`.
pub struct Device(pub mlx_rs::Device);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        for idx in 0..10 {
            let cpu = Device(mlx_rs::Device::new(mlx_rs::DeviceType::Cpu, idx));
            let gpu = Device(mlx_rs::Device::new(mlx_rs::DeviceType::Gpu, idx));

            assert!(cpu.0.get_index().is_ok_and(|id| { id == idx }));
            assert!(gpu.0.get_index().is_ok_and(|id| { id == idx }));
        }
    }
}
