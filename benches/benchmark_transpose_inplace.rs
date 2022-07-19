use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use transpose::ej::transpose_inplace;
use transpose::inplace::ip_transpose;

const SIZES: [usize; 8] = [56, 128, 256, 512, 1024, 2048, 4096, 8192];

pub fn bench_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - inplace");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.into_iter() {
        let m = n + 0;
        let mut src = vec![0.; n * m];
        let iw = m;
        let mut w = vec![0.; iw];
        let name = format!("Size: {} x {}", n, m);
        group.bench_function(&name, |b| b.iter(|| ip_transpose(&mut src, &mut w, n, m)));
    }
    group.finish();
}

pub fn bench_inplace_ej(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - inplace - EJ");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.into_iter() {
        let m = n + 0;
        let mut src = vec![0.; n * m];
        let iw = m;
        let mut w = vec![0.; iw];
        let name = format!("Size: {} x {}", n, m);
        group.bench_function(&name, |b| {
            b.iter(|| transpose_inplace(&mut src, &mut w, n, m))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_inplace, bench_inplace_ej);
// criterion_group!(benches2, bench_tile_ej2);
criterion_main!(benches);
