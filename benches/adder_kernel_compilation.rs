use criterion::{criterion_group, criterion_main, Criterion};
use metalogue::{Device, Kernel};

fn acquire_default_gpu(c: &mut Criterion) {
    c.bench_function("acquire_default_gpu", |b| {
        b.iter(|| {
            let _device =
                std::hint::black_box(Device::acquire().expect("failed to acquire GPU device"));
        });
    });
}

fn add_kernel_compilation(c: &mut Criterion) {
    let device = Device::acquire().expect("failed to acquire GPU device");
    let kernel_source = include_str!("../kernels/adder.metal");
    let kernel = Kernel::new(kernel_source, "add_arrays");

    c.bench_function("add_kernel_compilation", |b| {
        b.iter(|| {
            kernel
                .to_pipeline(&device)
                .expect("failed to compile kernel")
        })
    });
}

criterion_group!(benches, acquire_default_gpu, add_kernel_compilation);
criterion_main!(benches);
