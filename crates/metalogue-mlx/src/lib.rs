use mlx_rs;

/// A thin wrapper around `mlx_rs::Device`.
pub struct Device(pub mlx_rs::Device);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let cpu = Device(mlx_rs::Device::new(mlx_rs::DeviceType::Cpu, 0));
        let gpu = Device(mlx_rs::Device::new(mlx_rs::DeviceType::Gpu, 0));

        assert!(cpu.0.get_index().is_ok_and(|id| { id == 0 }));
        assert!(gpu.0.get_index().is_ok_and(|id| { id == 0 }));
    }
}
