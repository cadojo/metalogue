use criterion::{Criterion, criterion_group, criterion_main};
use metalogue::{Device, Kernel};
use std::path::Path;

fn add_kernel_compilation(c: &mut Criterion) {
    let device = Device::acquire().expect("failed to acquire GPU device");
    let kernel_path = Path::new("../kernels/adder.metal");
    let kernel =
        Kernel::from_file(kernel_path, "add_arrays").expect("failed to load kernel from file");

    c.bench_function("add_kernel_compilation", |b| {
        b.iter(|| kernel.compile(&device).expect("failed to compile kernel"))
    });
}

criterion_group!(benches, add_kernel_compilation);
criterion_main!(benches);
