use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use transpose::ej::transpose;
use transpose::oop_transpose;
use transpose::outofplace::oop_transpose_small;
use transpose::outofplace::par_oop_transpose_small;

// const SIZES: [usize; 8] = [56, 128, 256, 512, 1024, 2048, 4096, 8192];
const SIZES: [usize; 3] = [56, 128, 256];

pub fn bench_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - small");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.into_iter() {
        let src = vec![0.; n * n];
        let mut dst = vec![0.; n * n];
        let name = format!("Size: {} x {}", n, n);
        group.bench_function(&name, |b| {
            b.iter(|| oop_transpose_small(&src, &mut dst, n, n))
        });
    }
    group.finish();
}

pub fn bench_small2(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - small - par");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.into_iter() {
        let src = vec![0.; n * n];
        let mut dst = vec![0.; n * n];
        let name = format!("Size: {} x {}", n, n);
        group.bench_function(&name, |b| {
            b.iter(|| par_oop_transpose_small(&src, &mut dst, n, n))
        });
    }
    group.finish();
}

/*pub fn bench_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - inplace");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.into_iter() {
        let mut dst = vec![0.; n * n];
        let name = format!("Size: {} x {}", n, n);
        group.bench_function(&name, |b| {
            b.iter(|| unsafe { transpose_inplace(&mut dst, n, n) })
        });
    }
    group.finish();
}*/

// pub fn bench_tile_no1(c: &mut Criterion) {
//     let mut group = c.benchmark_group("transpose - tile #1");
//     group.significance_level(0.1).sample_size(10);
//     let block = 16;
//     for n in SIZES.into_iter() {
//         let src = vec![0.; n * n];
//         let mut dst = vec![0.; n * n];
//         let name = format!("Size: {} x {} | Block size {}", n, n, block);
//         group.bench_function(&name, |b| {
//             b.iter(|| unsafe { transpose_tiling_no1(&src, &mut dst, n, n, block) })
//         });
//     }
//     group.finish();
// }

pub fn bench_outofplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - out of place");
    group.significance_level(0.1).sample_size(10);
    // let block = 16;
    for n in SIZES.into_iter() {
        let src = vec![0.; n * n];
        let mut dst = vec![0.; n * n];
        let name = format!("Size: {} x {}", n, n);
        group.bench_function(&name, |b| b.iter(|| oop_transpose(&src, &mut dst, n, n)));
    }
    group.finish();
}

// pub fn bench_tiling(c: &mut Criterion) {
//     let mut group = c.benchmark_group("transpose - tile #2");
//     group.significance_level(0.1).sample_size(10);
//     let block = 16;
//     for n in SIZES.into_iter() {
//         let src = vec![0.; n * n];
//         let mut dst = vec![0.; n * n];
//         let name = format!("Size: {} x {} | Block size {}", n, n, block);
//         group.bench_function(&name, |b| {
//             b.iter(|| oop_transpose_tiling(&src, &mut dst, n, n, block))
//         });
//     }
//     group.finish();
// }

pub fn bench_ej(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose - EJ");
    group.significance_level(0.1).sample_size(10);
    // let block = 16;
    for n in SIZES.into_iter() {
        let src = vec![0.; n * n];
        let mut dst = vec![0.; n * n];
        let name = format!("Size: {} x {}", n, n);
        group.bench_function(&name, |b| b.iter(|| transpose(&src, &mut dst, n, n)));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_small,
    bench_small2,
    // bench_tile_no1,
    // bench_inplace,
    // bench_outofplace,
    // bench_ej
);
// criterion_group!(benches2, bench_tile_ej2);
criterion_main!(benches);
