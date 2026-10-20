//! The mass averaging of openEMS's `sar_calc` methods (and of its in-run
//! averaged SAR dumps), reproduced from black-box observations of
//! openEMS v0.37 (no openEMS source consulted; experiments on synthetic
//! raw SAR files, see `docs/DESIGN.md` §5.12 and
//! `scripts/openems-oracle/gen-sar.sh`).
//!
//! # Cube solver
//!
//! Per tissue voxel and cube placement the side `s` is found by a secant
//! iteration on the cube mass `M(s) − m`:
//!
//! 1. `s₀ = (m/ρ_v)^{1/3}` with the voxel's density, `a = s₀/2`;
//! 2. the mean density of the cube of side `a`, `ρ̄ = M(a)/a³` (the full
//!    geometric volume, also where the cube is clipped by the domain);
//! 3. the start `b = max((m/ρ̄)^{1/3}, 2·min(Δx, Δy, Δz))` (the voxel's
//!    widths);
//! 4. secant steps through the last two points, starting from `(a, M(a))`
//!    and `(b, M(b))`, until `|M − m| ≤ tol·m`: `tol` = 5 % for `SIMPLE`
//!    and `IEEE_C95_3`, 1e-6 for `IEEE_62704`. A step that cannot be
//!    taken (equal masses, a non-positive side) or 100 steps leave the
//!    voxel without a cube.
//!
//! `M`, the power and the cube volume weight every voxel with the product
//! of its per-axis overlap fractions times its volume (the raw file's
//! `Volume` dataset in `sar_calc`); outside the domain counts as nothing,
//! air (`ρ = 0`) as no mass and no power. Observed quirk, reproduced: a
//! voxel that the cube cuts on both sides along an axis (the cube lies
//! inside it along that axis) gets the square of that fraction.
//!
//! # Methods
//!
//! * `SIMPLE`: every voxel with a cube gets its average.
//! * `IEEE_C95_3`, `IEEE_62704`: a centred cube is *valid* when it lies in
//!   the domain and each of its six faces touches tissue (some tissue
//!   voxel in the voxel layer the face lies in, over the face's extent;
//!   an air pocket inside is allowed); for `IEEE_62704` also at most 10 %
//!   of its volume `s³` may be air. A voxel without a valid cube takes the
//!   largest average of the valid cubes that overlap it (C95.3) or contain
//!   it entirely (62704); a voxel reached by none takes one of the six
//!   cubes with the voxel at the centre of a face (the cube extending from
//!   one of the voxel's faces, centred across): the one with the smallest
//!   volume within the domain (C95.3), or of those whose volume is at most
//!   1.05 times the smallest, the one with the largest average (62704).
//!
//! # Known differences
//!
//! Where openEMS's choice hinges on rounding, ems decides with its own:
//! a face exactly on a tissue boundary (both codes then see the cube
//! slightly larger than exact, through the single-precision cell volumes,
//! and call the face in air), C95.3 face cubes of equal volume (ems takes
//! the largest average), and an iterate at the edge of the 62704
//! tolerance. Not reproduced: cubes of less than about 0.85 cell (the
//! squared fractions make the mass non-monotonic), and the IEEE validity
//! of some cubes of about two cells on graded meshes (openEMS rejects a
//! few whose faces touch tissue).

use super::{AveragingMethod, SarField, SarVolume, VoxelClass};

/// Largest number of secant steps.
const MAX_STEPS: usize = 100;

/// Prefix sums of per-cell `[ρ·V, p·V (tissue), V, tissue, V (tissue)]`.
struct Prefix {
    n: [usize; 3],
    data: Vec<[f64; 5]>,
}

impl Prefix {
    fn new(c: [usize; 3], cell: impl Fn(usize) -> [f64; 5]) -> Self {
        let n = [c[0] + 1, c[1] + 1, c[2] + 1];
        let idx = |i: usize, j: usize, k: usize| (i * n[1] + j) * n[2] + k;
        let mut data = vec![[0.0; 5]; n[0] * n[1] * n[2]];
        for i in 1..n[0] {
            for j in 1..n[1] {
                for k in 1..n[2] {
                    let v = cell(((i - 1) * c[1] + j - 1) * c[2] + k - 1);
                    let mut s = [0.0; 5];
                    for (q, s) in s.iter_mut().enumerate() {
                        *s = v[q]
                            + data[idx(i - 1, j, k)][q]
                            + data[idx(i, j - 1, k)][q]
                            + data[idx(i, j, k - 1)][q]
                            - data[idx(i - 1, j - 1, k)][q]
                            - data[idx(i - 1, j, k - 1)][q]
                            - data[idx(i, j - 1, k - 1)][q]
                            + data[idx(i - 1, j - 1, k - 1)][q];
                    }
                    data[idx(i, j, k)] = s;
                }
            }
        }
        Self { n, data }
    }

    /// Sum over the cells `r[0] × r[1] × r[2]` (half-open ranges).
    fn sum(&self, r: [(usize, usize); 3]) -> [f64; 5] {
        let n = self.n;
        let at = |i: usize, j: usize, k: usize| &self.data[(i * n[1] + j) * n[2] + k];
        let ((i0, i1), (j0, j1), (k0, k1)) = (r[0], r[1], r[2]);
        let mut s = [0.0; 5];
        for (q, s) in s.iter_mut().enumerate() {
            *s = at(i1, j1, k1)[q] - at(i0, j1, k1)[q] - at(i1, j0, k1)[q] - at(i1, j1, k0)[q]
                + at(i0, j0, k1)[q]
                + at(i0, j1, k0)[q]
                + at(i1, j0, k0)[q]
                - at(i0, j0, k0)[q];
        }
        s
    }
}

/// Up to three `(cell range, weight)` pairs covering `[a, b]` on one axis
/// (clipped to the domain), the weight being the covered fraction of each
/// cell, squared for a cell cut on both sides.
type Segments = ([((usize, usize), f64); 3], usize);

fn segments(lines: &[f64], a: f64, b: f64) -> Segments {
    let mut out = [((0, 0), 0.0); 3];
    let n = lines.len() - 1;
    let (a, b) = (a.max(lines[0]), b.min(lines[n]));
    if b.is_nan() || a.is_nan() || b <= a {
        return (out, 0);
    }
    let i0 = lines.partition_point(|&x| x <= a).clamp(1, n) - 1;
    let i1 = lines.partition_point(|&x| x < b).clamp(1, n) - 1;
    let d = |i: usize| lines[i + 1] - lines[i];
    if i0 == i1 {
        let f = (b - a) / d(i0);
        let w = if a > lines[i0] && b < lines[i0 + 1] {
            f * f
        } else {
            f
        };
        out[0] = ((i0, i0 + 1), w);
        return (out, 1);
    }
    let mut m = 0;
    out[m] = ((i0, i0 + 1), (lines[i0 + 1] - a) / d(i0));
    m += 1;
    if i1 > i0 + 1 {
        out[m] = ((i0 + 1, i1), 1.0);
        m += 1;
    }
    out[m] = ((i1, i1 + 1), (b - lines[i1]) / d(i1));
    (out, m + 1)
}

/// Where a cube sits relative to its voxel: centred, or extending from
/// one of the voxel's faces (openEMS cube type codes 1…6: `+x` from the
/// lower x face, `−x` from the upper one, then y, z).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    Centered,
    /// openEMS code 1…6.
    Face(u8),
}

/// A solved cube.
#[derive(Debug, Clone, Copy)]
struct Cube {
    lo: [f64; 3],
    hi: [f64; 3],
    sar: f64,
    mass: f64,
    volume: f64,
    /// Non-tissue fraction of the cube's full volume `s³`.
    air: f64,
}

struct Averager<'a> {
    v: &'a SarVolume,
    mass: f64,
    method: AveragingMethod,
    tol: f64,
    threads: usize,
    c: [usize; 3],
    prefix: Prefix,
    centers: [Vec<f64>; 3],
}

/// See the [module docs](self); `volumes` per cell.
pub(super) fn average(
    v: &SarVolume,
    volumes: &[f64],
    mass: f64,
    method: AveragingMethod,
    threads: usize,
) -> SarField {
    let c = v.cells();
    let prefix = Prefix::new(c, |i| {
        let (r, vol) = (v.density[i], volumes[i]);
        if r > 0.0 {
            [r * vol, v.power[i] * vol, vol, 1.0, vol]
        } else {
            [0.0, 0.0, vol, 0.0, 0.0]
        }
    });
    let centers =
        core::array::from_fn(|a| v.lines[a].windows(2).map(|w| 0.5 * (w[0] + w[1])).collect());
    let a = Averager {
        v,
        mass,
        method,
        tol: if method == AveragingMethod::Ieee62704 {
            1e-6
        } else {
            0.05
        },
        threads: match threads {
            0 => std::thread::available_parallelism().map_or(1, usize::from),
            n => n,
        },
        c,
        prefix,
        centers,
    };
    a.run()
}

impl Averager<'_> {
    fn cell(&self, i: usize) -> [usize; 3] {
        let c = self.c;
        [i / (c[1] * c[2]), (i / c[2]) % c[1], i % c[2]]
    }

    fn index(&self, p: [usize; 3]) -> usize {
        (p[0] * self.c[1] + p[1]) * self.c[2] + p[2]
    }

    /// `[mass, tissue power, volume, -, tissue volume]` in the box
    /// `lo..hi`.
    fn integrate(&self, lo: [f64; 3], hi: [f64; 3]) -> [f64; 5] {
        let s: [_; 3] = core::array::from_fn(|a| segments(&self.v.lines[a], lo[a], hi[a]));
        let mut out = [0.0; 5];
        for &(rx, wx) in &s[0].0[..s[0].1] {
            for &(ry, wy) in &s[1].0[..s[1].1] {
                for &(rz, wz) in &s[2].0[..s[2].1] {
                    let w = wx * wy * wz;
                    let v = self.prefix.sum([rx, ry, rz]);
                    for q in 0..5 {
                        out[q] += w * v[q];
                    }
                }
            }
        }
        out
    }

    fn bounds(&self, cell: [usize; 3], p: Placement, s: f64) -> ([f64; 3], [f64; 3]) {
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for a in 0..3 {
            let l = &self.v.lines[a];
            let (x0, x1) = (l[cell[a]], l[cell[a] + 1]);
            (lo[a], hi[a]) = match p {
                Placement::Face(k) if usize::from((k - 1) / 2) == a => {
                    if k % 2 == 1 {
                        (x0, x0 + s)
                    } else {
                        (x1 - s, x1)
                    }
                }
                _ => {
                    let x = self.centers[a][cell[a]];
                    (x - 0.5 * s, x + 0.5 * s)
                }
            };
        }
        (lo, hi)
    }

    /// The cube of mass `self.mass` at `cell` with placement `p`, if the
    /// solver finds one.
    fn cube(&self, cell: [usize; 3], p: Placement) -> Option<Cube> {
        let m = self.mass;
        let i = self.index(cell);
        let rv = self.v.density[i];
        let eval = |s: f64| {
            let (lo, hi) = self.bounds(cell, p, s);
            self.integrate(lo, hi)
        };
        let s0 = (m / rv).cbrt();
        let mut a = 0.5 * s0;
        let ma = eval(a)[0];
        let rbar = ma / (a * a * a);
        if !(rbar > 0.0 && rbar.is_finite()) {
            return None;
        }
        let min_side = 2.0
            * (0..3)
                .map(|ax| self.v.lines[ax][cell[ax] + 1] - self.v.lines[ax][cell[ax]])
                .fold(f64::INFINITY, f64::min);
        let mut b = (m / rbar).cbrt().max(min_side);
        let mut fa = ma - m;
        let mut q = eval(b);
        let mut fb = q[0] - m;
        let mut steps = 0;
        while fb.abs() > self.tol * m {
            if steps == MAX_STEPS || fb == fa {
                return None;
            }
            let x = b - fb * (b - a) / (fb - fa);
            if !(x > 0.0 && x.is_finite()) {
                return None;
            }
            (a, fa, b) = (b, fb, x);
            q = eval(b);
            fb = q[0] - m;
            steps += 1;
        }
        if q[0].is_nan() || q[0] <= 0.0 {
            return None;
        }
        let (lo, hi) = self.bounds(cell, p, b);
        Some(Cube {
            lo,
            hi,
            sar: q[1] / q[0],
            mass: q[0],
            volume: q[2],
            air: 1.0 - q[4] / (b * b * b),
        })
    }

    /// Cells with a positive overlap with `[lo, hi]` along axis `a`
    /// (inclusive), within the domain.
    fn range(&self, a: usize, lo: f64, hi: f64) -> (usize, usize) {
        let l = &self.v.lines[a];
        let n = self.c[a];
        (
            l.partition_point(|&x| x <= lo).clamp(1, n) - 1,
            l.partition_point(|&x| x < hi).clamp(1, n) - 1,
        )
    }

    /// The IEEE validity of a centred cube: inside the domain, every face
    /// touching tissue, and (62704) at most 10 % air.
    fn valid(&self, q: &Cube) -> bool {
        if self.method == AveragingMethod::Ieee62704 && q.air > 0.1 {
            return false;
        }
        for a in 0..3 {
            let l = &self.v.lines[a];
            if q.lo[a] < l[0] || q.hi[a] > l[self.c[a]] {
                return false;
            }
        }
        let r: [(usize, usize); 3] = core::array::from_fn(|a| self.range(a, q.lo[a], q.hi[a]));
        for a in 0..3 {
            for layer in [r[a].0, r[a].1] {
                let mut box_ = r.map(|(i0, i1)| (i0, i1 + 1));
                box_[a] = (layer, layer + 1);
                if self.prefix.sum(box_)[3] < 0.5 {
                    return false;
                }
            }
        }
        true
    }

    /// `f(i)` for every `i < n`, in parallel.
    fn par_map<T: Send + Clone + Default>(
        &self,
        n: usize,
        f: impl Fn(usize) -> T + Sync,
    ) -> Vec<T> {
        let t = self.threads.clamp(1, n.max(1));
        let mut out = vec![T::default(); n];
        let chunk = n.div_ceil(t).max(1);
        std::thread::scope(|sc| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                let f = &f;
                sc.spawn(move || {
                    for (o, slot) in part.iter_mut().enumerate() {
                        *slot = f(c * chunk + o);
                    }
                });
            }
        });
        out
    }

    /// The largest average of the valid cubes reaching `cell` (overlap for
    /// C95.3, containment for 62704).
    fn covered(&self, cell: [usize; 3], valid: &[Option<Cube>], half: f64) -> Option<f64> {
        let l = &self.v.lines;
        let vlo = [0, 1, 2].map(|a| l[a][cell[a]]);
        let vhi = [0, 1, 2].map(|a| l[a][cell[a] + 1]);
        let window = |a: usize| {
            let c = &self.centers[a];
            c.partition_point(|&y| y < vlo[a] - half)..c.partition_point(|&y| y <= vhi[a] + half)
        };
        let overlap = self.method == AveragingMethod::IeeeC95_3;
        let mut best: Option<f64> = None;
        for p in window(0) {
            for q in window(1) {
                for r in window(2) {
                    let Some(cube) = &valid[self.index([p, q, r])] else {
                        continue;
                    };
                    let reaches = (0..3).all(|a| {
                        if overlap {
                            cube.lo[a] < vhi[a] && cube.hi[a] > vlo[a]
                        } else {
                            cube.lo[a] <= vlo[a] && cube.hi[a] >= vhi[a]
                        }
                    });
                    if reaches && best.is_none_or(|b| cube.sar > b) {
                        best = Some(cube.sar);
                    }
                }
            }
        }
        best
    }

    /// The face cube of a voxel (code, cube).
    fn face_cube(&self, cell: [usize; 3]) -> Option<(u8, Cube)> {
        let cubes: Vec<(u8, Cube)> = (1..=6u8)
            .filter_map(|k| self.cube(cell, Placement::Face(k)).map(|q| (k, q)))
            .collect();
        let vmin = cubes
            .iter()
            .map(|(_, q)| q.volume)
            .fold(f64::INFINITY, f64::min);
        // 62704: the largest average within 5 % of the smallest volume;
        // C95.3: the smallest volume (equal volumes, which openEMS orders
        // by its rounding, by the largest average).
        let band = if self.method == AveragingMethod::Ieee62704 {
            1.05
        } else {
            1.0 + 1e-12
        };
        let mut pick: Option<(u8, Cube)> = None;
        for &(k, q) in &cubes {
            if q.volume <= band * vmin && pick.is_none_or(|(_, b)| q.sar > b.sar) {
                pick = Some((k, q));
            }
        }
        pick
    }

    fn run(&self) -> SarField {
        let n: usize = self.c.iter().product();
        let tissue = |i: usize| self.v.density[i] > 0.0;
        let ieee = self.method != AveragingMethod::Simple;
        let centred: Vec<Option<Cube>> = self.par_map(n, |i| {
            if !tissue(i) {
                return None;
            }
            self.cube(self.cell(i), Placement::Centered)
                .filter(|q| !ieee || self.valid(q))
        });
        let half = centred
            .iter()
            .flatten()
            .map(|q| {
                (0..3)
                    .map(|a| 0.5 * (q.hi[a] - q.lo[a]))
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        // (value, class, face code, cube mass, cube volume)
        type Out = (f64, VoxelClass, u8, f64, f64);
        let rest: Vec<Out> = self.par_map(n, |i| {
            if !tissue(i) {
                return (0.0, VoxelClass::NoTissue, 0, 0.0, 0.0);
            }
            if let Some(q) = centred[i] {
                return (q.sar, VoxelClass::Centered, 0, q.mass, q.volume);
            }
            if !ieee {
                return (0.0, VoxelClass::Unresolved, 0, 0.0, 0.0);
            }
            let cell = self.cell(i);
            if let Some(s) = self.covered(cell, &centred, half) {
                return (s, VoxelClass::Covered, 0, 0.0, 0.0);
            }
            match self.face_cube(cell) {
                Some((k, q)) => (q.sar, VoxelClass::Boundary, k, q.mass, q.volume),
                None => (0.0, VoxelClass::Unresolved, 0, 0.0, 0.0),
            }
        });
        let mut values = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(n);
        let mut faces = Vec::with_capacity(n);
        let mut cube_mass = Vec::with_capacity(n);
        let mut cube_volume = Vec::with_capacity(n);
        for (v, c, k, m, vol) in rest {
            values.push(v);
            classes.push(c);
            faces.push(k);
            cube_mass.push(m);
            cube_volume.push(vol);
        }
        SarField {
            lines: self.v.lines.clone(),
            values,
            classes,
            mass: Some(self.mass),
            cube_mass,
            cube_volume,
            cube_face: faces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n³` cells of 1 mm, tissue (1000 kg/m³) where `tissue(i, j, k)`,
    /// power density `p(i, j, k)`.
    fn volume(
        n: usize,
        tissue: impl Fn(usize, usize, usize) -> bool,
        p: impl Fn(usize, usize, usize) -> f64,
    ) -> SarVolume {
        let lines = [(); 3].map(|_| (0..=n).map(|i| i as f64 * 1e-3).collect::<Vec<_>>());
        let (mut power, mut density) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let t = tissue(i, j, k);
                    power.push(if t { p(i, j, k) } else { 0.0 });
                    density.push(if t { 1000.0 } else { 0.0 });
                }
            }
        }
        SarVolume::new(lines, power, density).unwrap()
    }

    #[test]
    fn simple_constant_field_is_local_everywhere() {
        let v = volume(10, |i, _, _| i >= 2, |_, _, _| 3.0);
        // 64 mm³ cubes: clipped by air and by the domain at the edges.
        let f = v
            .averaged_openems(64e-9 * 1000.0, AveragingMethod::Simple, 2)
            .unwrap();
        for (i, (&s, &c)) in f.values().iter().zip(f.classes()).enumerate() {
            if v.density()[i] > 0.0 {
                assert_eq!(c, VoxelClass::Centered);
                assert!((s - 3e-3).abs() < 1e-15, "{i}: {s}");
                // Solved to 5 % of the mass.
                assert!((f.cube_masses()[i] / 64e-6 - 1.0).abs() <= 0.05);
            }
        }
    }

    #[test]
    fn ieee_validity_coverage_and_face_cubes() {
        // A 6³ tissue block in air, a gradient along x, cubes of 2.6 mm.
        let v = volume(
            10,
            |i, j, k| [i, j, k].iter().all(|&x| (2..8).contains(&x)),
            |i, _, _| 1.0 + i as f64,
        );
        let m = 2.6e-3f64.powi(3) * 1000.0;
        let idx = |i: usize, j: usize, k: usize| (i * 10 + j) * 10 + k;
        let c95 = v
            .averaged_openems(m, AveragingMethod::IeeeC95_3, 1)
            .unwrap();
        let p62 = v
            .averaged_openems(m, AveragingMethod::Ieee62704, 3)
            .unwrap();
        for f in [&c95, &p62] {
            // Inner cubes (centres 3..=6) lie in tissue; the outer layer's
            // cubes have a face in air.
            assert_eq!(f.classes()[idx(4, 4, 4)], VoxelClass::Centered);
            assert_eq!(f.classes()[idx(0, 4, 4)], VoxelClass::NoTissue);
        }
        // C95.3: the edge voxel (2, ·) takes the largest overlapping valid
        // cube, the one centred at 3 (x 2.2..4.8: SAR 4e-3 W/kg).
        assert_eq!(c95.classes()[idx(2, 4, 4)], VoxelClass::Covered);
        assert!((c95.values()[idx(2, 4, 4)] - 4e-3).abs() < 1e-12);
        // 62704 requires containment: no valid cube contains it, it takes
        // the smallest face cube, extending along +x from its lower face
        // (x 2..4.6, all tissue).
        assert_eq!(p62.classes()[idx(2, 4, 4)], VoxelClass::Boundary);
        assert_eq!(p62.cube_faces()[idx(2, 4, 4)], 1);
        assert!((p62.values()[idx(2, 4, 4)] - 1e-3 * 10.0 / 2.6).abs() < 1e-9);
        assert_eq!(p62.classes()[idx(3, 4, 4)], VoxelClass::Centered);
        // Same result with any thread count.
        let again = v
            .averaged_openems(m, AveragingMethod::Ieee62704, 1)
            .unwrap();
        assert_eq!(again, p62);
    }

    #[test]
    fn thin_slab_takes_face_cubes() {
        // One cell thick: no valid cube; every voxel takes a face cube
        // extending along y or z (the slab plane).
        let v = volume(10, |i, _, _| i == 5, |_, _, _| 2.0);
        let f = v
            .averaged_openems(8e-9 * 1000.0, AveragingMethod::Ieee62704, 0)
            .unwrap();
        let mut faces = 0;
        for (i, &c) in f.classes().iter().enumerate() {
            if v.density()[i] > 0.0 {
                assert!(
                    matches!(c, VoxelClass::Boundary | VoxelClass::Unresolved),
                    "{i}: {c:?}"
                );
                if c == VoxelClass::Boundary {
                    faces += 1;
                    assert!((1..=6).contains(&f.cube_faces()[i]));
                    assert!((f.values()[i] - 2e-3).abs() < 1e-12);
                }
            }
        }
        assert!(faces > 0);
    }

    #[test]
    fn segments_square_a_cell_cut_on_both_sides() {
        let l = [0.0, 1.0, 2.0, 3.0];
        let (s, n) = segments(&l, 1.25, 1.75);
        assert_eq!(n, 1);
        assert_eq!(s[0], ((1, 2), 0.25));
        // Touching a line is a single cut.
        let (s, _) = segments(&l, 1.0, 1.5);
        assert_eq!(s[0], ((1, 2), 0.5));
        let (s, n) = segments(&l, 0.5, 2.25);
        assert_eq!(n, 3);
        assert_eq!(s[..3], [((0, 1), 0.5), ((1, 2), 1.0), ((2, 3), 0.25)]);
    }
}
