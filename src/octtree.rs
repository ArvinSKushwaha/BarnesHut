use ultraviolet::{Vec3, Vec3x8, f32x8};

const EPSILON: f32 = 1e-7;

#[derive(Clone, Debug)]
pub struct Octree {
    point: ([usize; 8], Vec3x8, f32x8),
    count: u8,
    com: Vec3,
    total_mass: f32,
    children: Option<Box<[Octree; 8]>>,
    center: Vec3,
    extent: Vec3,
}

impl Octree {
    pub fn construct(points: &[Vec3], masses: &[f32]) -> Self {
        let mut octree = Octree::default();

        assert_eq!(
            points.len(),
            masses.len(),
            "Length of given points not equal to length of given masses"
        );
        if points.len() == 0 {
            return octree;
        }

        let min_bound = points
            .iter()
            .copied()
            .reduce(Vec3::min_by_component)
            .unwrap();
        let max_bound = points
            .iter()
            .copied()
            .reduce(Vec3::max_by_component)
            .unwrap();

        octree.center = (min_bound + max_bound) / 2.0;
        octree.extent = max_bound - min_bound;

        // octree.center -= octree.extent * EPSILON;
        // octree.extent += octree.extent * 2. * EPSILON;

        points
            .iter()
            .zip(masses)
            .enumerate()
            .for_each(|(i, (&point, &mass))| {
                octree.add_point(i, point, mass);
            });

        octree.compute();
        octree
    }

    pub fn add_point(&mut self, idx: usize, point: Vec3, mass: f32) {
        if self.count < 8 {
            let count = self.count as usize;

            self.point.0[count] = idx;

            let mut x_array = self.point.1.x.to_array();
            let mut y_array = self.point.1.y.to_array();
            let mut z_array = self.point.1.z.to_array();
            let mut mass_array = self.point.2.to_array();

            x_array[count] = point.x;
            y_array[count] = point.y;
            z_array[count] = point.z;
            mass_array[count] = mass;

            self.point.1.x = f32x8::new(x_array);
            self.point.1.y = f32x8::new(y_array);
            self.point.1.z = f32x8::new(z_array);
            self.point.2 = f32x8::new(mass_array);

            self.count += 1;
        } else {
            let child_idx = {
                let diff = point - self.center;
                (diff.x.is_sign_positive() as usize) << 2
                    | (diff.y.is_sign_positive() as usize) << 1
                    | (diff.z.is_sign_positive() as usize)
            };

            self.get_child(child_idx).add_point(idx, point, mass);
        }
    }

    pub fn get_child(&mut self, idx: usize) -> &mut Octree {
        if let Some(ref mut children) = self.children {
            &mut children[idx]
        } else {
            let offsets = [
                Vec3::new(-1., -1., -1.),
                Vec3::new(-1., -1., 1.),
                Vec3::new(-1., 1., -1.),
                Vec3::new(-1., 1., 1.),
                Vec3::new(1., -1., -1.),
                Vec3::new(1., -1., 1.),
                Vec3::new(1., 1., -1.),
                Vec3::new(1., 1., 1.),
            ];

            let mut children: [Octree; 8] = Default::default();

            children.iter_mut().enumerate().for_each(|(i, child)| {
                child.extent = self.extent / 2.0;
                child.center = self.center + self.extent * offsets[i] / 4.0;
            });

            self.children = Some(Box::new(children));
            &mut self.children.as_mut().unwrap()[idx]
        }
    }

    pub fn compute(&mut self) {
        let (mut com, mut total_mass) = (Vec3::zero(), 0.);

        let (_, self_com, self_mass) = self.point;
        com += {
            let tmp = self_com * self_mass;
            Vec3::new(tmp.x.reduce_add(), tmp.y.reduce_add(), tmp.z.reduce_add())
        };
        total_mass += self_mass.reduce_add();

        if let Some(ref mut children) = self.children {
            children.iter_mut().for_each(Octree::compute);

            let (child_com, child_mass) = children.iter().fold((Vec3::zero(), 0.), |a, b| {
                (a.0 + b.total_mass * b.com, a.1 + b.total_mass)
            });

            com += child_com;
            total_mass += child_mass;
        }

        self.total_mass = total_mass;
        self.com = if total_mass == 0. {
            Vec3::zero()
        } else {
            com / total_mass
        };
    }

    pub fn find(&self, idx: usize) -> Option<&Octree> {
        if self.point.0.contains(&idx) {
            return Some(&self);
        }

        self.children
            .as_ref()?
            .iter()
            .find_map(|tree| tree.find(idx))
    }
}

impl Default for Octree {
    fn default() -> Self {
        Self {
            point: ([0; 8], Vec3x8::zero(), f32x8::ZERO),
            count: 0,
            total_mass: 0.0,
            com: Vec3::zero(),
            children: None,
            center: Vec3::zero(),
            extent: Vec3::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Octree;
    use ultraviolet::Vec3;

    #[test]
    fn test_octree_correctness() {
        let points = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let masses = vec![1.; 8];

        let oct = Octree::construct(&points, &masses);
        println!("{:#?}", oct);

        assert_eq!(oct.total_mass, 8.);
        assert_eq!(oct.com, Vec3::zero());

        (0..8).for_each(|i| {
            let found = oct.find(i).expect("Could not find element");
            let disp = points[i] - found.center;
            let half_extent = found.extent / 2.0;

            assert!(
                -half_extent.x <= disp.x && disp.x <= half_extent.x,
                "{:?} not in {:?} with half_extent {:?}",
                points[i],
                found.center,
                found.extent
            );
            assert!(
                -half_extent.y <= disp.y && disp.y <= half_extent.y,
                "{:?} not in {:?} with half_extent {:?}",
                points[i],
                found.center,
                found.extent
            );
            assert!(
                -half_extent.z <= disp.z && disp.z <= half_extent.z,
                "{:?} not in {:?} with half_extent {:?}",
                points[i],
                found.center,
                found.extent
            );
        });
    }
}
