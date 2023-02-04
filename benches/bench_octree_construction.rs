use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{prelude::*, Fill};
use ultraviolet::Vec3;

use barnes_hut::octtree::Octree;
fn bench_octree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Octree Construction");
    group.measurement_time(Duration::from_secs(60));

    for size in (4usize..=20usize).map(|v| 1usize << v) {
        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Octree Construction", size),
            &size,
            |b, &size| {
                let mut points: Vec<Vec3> = vec![Vec3::zero(); size];
                let mut masses: Vec<f32> = vec![0.; size];

                points
                    .iter_mut()
                    .map(Vec3::as_mut_slice)
                    .for_each(|s| s.copy_from_slice(&rand::thread_rng().gen::<[f32; 3]>()));

                masses
                    .try_fill(&mut rand::thread_rng())
                    .expect("Failed to fill masses");

                b.iter(|| Octree::construct(black_box(&points), black_box(&masses)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_octree_construction);
criterion_main!(benches);
