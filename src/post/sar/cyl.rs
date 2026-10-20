//! Mass-averaged SAR on cylindrical cells (design §5.12, §5.15).
//!
//! # Physical cubes (ems)
//!
//! [`SarVolume::averaged_with`] on [cylindrical](SarVolume::cylindrical)
//! cells averages over cubes in **physical** space: axis-aligned with the
//! Cartesian `x, y, z` axes (`x = ρ·cos α`, `y = ρ·sin α`), centred on the
//! voxel centre `(ρ_{i+½}, α_{j+½}, z_{k+½})`, and grown until their
//! tissue mass is `m`, with the same three-step procedure (centred cubes,
//! covered voxels, face cubes) and the same air rule as on Cartesian cells
//! (see the [parent module](super#mass-averaging)).
//!
//! The overlap of a cube with an annular-sector cell
//! `[ρ_0, ρ_1] × [α_0, α_1] × [z_0, z_1]` is exact: the z overlap is an
//! interval, and the area of the square `Q` (the cube's xy section)
//! inside the sector is `|Q ∩ W ∩ D(ρ_1)| − |Q ∩ W ∩ D(ρ_0)|`, with `W` the
//! wedge `α_0 ≤ α ≤ α_1` (two half-planes, cells wider than π/2 are split)
//! and `D(r)` the disc of radius `r`. `Q ∩ W` is a convex polygon
//! (Sutherland–Hodgman clipping), and the area of a polygon inside a disc
//! centred on the origin is the sum, over its edges, of the signed area of
//! the triangle (origin, edge) inside the disc: straight pieces inside,
//! circular sectors `r²·Δθ/2` outside. Every cell then counts with its
//! exact volume fraction, so the cube mass is a continuous, increasing
//! function of the side, solved to 1e-12 of `m` like the Cartesian cubes.
//! A cube reaching outside the grid's region (beyond `ρ_max`, outside a
//! sector, beyond the z range) counts the outside as air.
//!
//! Face cubes put the voxel centre at the centre of one cube face (six
//! orientations, `±x`, `±y`, `±z`), since an annular cell has no face
//! normal to `x` or `y`.
//!
//! # openEMS's cubes
//!
//! openEMS (v0.37, observed as a black box) averages cylindrical raw SAR
//! data as if the `(ρ, α, z)` cells were Cartesian cells of widths
//! `Δρ`, `Δα·u` (`u` the drawing unit: the α widths in radians scaled like
//! lengths) and `Δz`, each cell weighing with its `Volume`
//! `ρ_i·Δα·Δρ·Δz` (`ρ_i` the cell's lower radius): its in-run 1 g dumps and
//! `sar_calc` on the cylindrical raw dump are bit-identical, and equal to
//! `sar_calc` on the same file with the axes renamed to `x, y, z`. Its
//! cubes are therefore boxes of `s/Δα·u` α cells, which are neither cubes
//! nor of constant physical size. For openEMS-compatible files
//! (`io::openems::dump`, `ems-sar-calc`) ems builds that Cartesian view
//! (the widths stacked from 0, [`SarVolume::with_cell_volumes`]) and
//! averages it with [`SarVolume::averaged_openems`].

use super::{AveragingOptions, Cube, Placement, SarField, SarVolume, VoxelClass, segments};

type P = [f64; 2];

fn cross(a: P, b: P) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Signed area of the triangle `(0, a, b)` inside the disc `|p| ≤ r`.
fn triangle_in_disc(a: P, b: P, r: f64) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let qa = dot(d, d);
    let r2 = r * r;
    let piece = |t0: f64, t1: f64| {
        let p = [a[0] + t0 * d[0], a[1] + t0 * d[1]];
        let q = [a[0] + t1 * d[0], a[1] + t1 * d[1]];
        let tm = 0.5 * (t0 + t1);
        let m = [a[0] + tm * d[0], a[1] + tm * d[1]];
        if dot(m, m) <= r2 {
            0.5 * cross(p, q)
        } else {
            0.5 * r2 * cross(p, q).atan2(dot(p, q))
        }
    };
    if qa == 0.0 {
        return 0.0;
    }
    let qb = dot(a, d);
    let qc = dot(a, a) - r2;
    let disc = qb * qb - qa * qc;
    let mut ts = [0.0, 1.0, 1.0, 1.0];
    let mut n = 1;
    if disc > 0.0 {
        let s = disc.sqrt();
        for t in [(-qb - s) / qa, (-qb + s) / qa] {
            if t > 0.0 && t < 1.0 {
                ts[n] = t;
                n += 1;
            }
        }
    }
    ts[n] = 1.0;
    (0..n).map(|i| piece(ts[i], ts[i + 1])).sum()
}

/// Area of the convex polygon `poly` (counter-clockwise) inside the disc
/// of radius `r` centred on the origin.
fn polygon_in_disc(poly: &[P], r: f64) -> f64 {
    if r <= 0.0 || poly.len() < 3 {
        return 0.0;
    }
    (0..poly.len())
        .map(|i| triangle_in_disc(poly[i], poly[(i + 1) % poly.len()], r))
        .sum::<f64>()
        .max(0.0)
}

/// Clips `poly` to the half-plane `cross(d, p) ≥ 0` (left of the line
/// through the origin along `d`).
fn clip_left(poly: &[P], d: P) -> Vec<P> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (sp, sq) = (cross(d, p), cross(d, q));
        if sp >= 0.0 {
            out.push(p);
        }
        if (sp >= 0.0) != (sq >= 0.0) {
            let t = sp / (sp - sq);
            out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
        }
    }
    out
}

/// Area of the rectangle `[x0, x1] × [y0, y1]` inside the annular sector
/// `r0 ≤ ρ ≤ r1`, `a0 ≤ α ≤ a1` (`a1 − a0 ≤ π/2`).
fn rect_in_sector(rect: [f64; 4], r0: f64, r1: f64, a0: f64, a1: f64) -> f64 {
    let [x0, y0, x1, y1] = rect;
    let sq = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let q = clip_left(&sq, [a0.cos(), a0.sin()]);
    if q.len() < 3 {
        return 0.0;
    }
    let d1 = [a1.cos(), a1.sin()];
    let q = clip_left(&q, [-d1[0], -d1[1]]);
    if q.len() < 3 {
        return 0.0;
    }
    (polygon_in_disc(&q, r1) - polygon_in_disc(&q, r0)).max(0.0)
}

/// Prefix sums along z per `(ρ, α)` column of `[mass, power, tissue
/// length, length]` per unit cross-section.
struct Columns {
    nz: usize,
    data: Vec<[f64; 4]>,
}

impl Columns {
    fn sum(&self, col: usize, r: (usize, usize)) -> [f64; 4] {
        let base = col * (self.nz + 1);
        let (a, b) = (&self.data[base + r.0], &self.data[base + r.1]);
        [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]]
    }
}

struct CylAverager<'a> {
    v: &'a SarVolume,
    mass: f64,
    opts: AveragingOptions,
    c: [usize; 3],
    cols: Columns,
    /// Per α cell, the sub-wedges (≤ π/2) it is split into.
    wedges: Vec<Vec<(f64, f64)>>,
    /// Physical voxel centres `[x, y, z]`.
    centres: Vec<[f64; 3]>,
    total_mass: f64,
    /// Bound on any useful cube side \[m\].
    s_max: f64,
}

impl<'a> CylAverager<'a> {
    fn new(v: &'a SarVolume, mass: f64, opts: &AveragingOptions) -> Self {
        let c = v.cells();
        let [r, a, z] = [&v.lines[0], &v.lines[1], &v.lines[2]];
        let mut data = vec![[0.0; 4]; c[0] * c[1] * (c[2] + 1)];
        for i in 0..c[0] {
            for j in 0..c[1] {
                let col = i * c[1] + j;
                let base = col * (c[2] + 1);
                for k in 0..c[2] {
                    let cell = col * c[2] + k;
                    let dz = z[k + 1] - z[k];
                    let rho = v.density[cell];
                    let add = if rho > 0.0 {
                        [rho * dz, v.power[cell] * dz, dz, dz]
                    } else {
                        [0.0, 0.0, 0.0, dz]
                    };
                    let prev = data[base + k];
                    data[base + k + 1] = core::array::from_fn(|q| prev[q] + add[q]);
                }
            }
        }
        let wedges = a
            .windows(2)
            .map(|w| {
                let n = ((w[1] - w[0]) / core::f64::consts::FRAC_PI_2)
                    .ceil()
                    .max(1.0) as usize;
                let d = (w[1] - w[0]) / n as f64;
                (0..n)
                    .map(|m| (w[0] + m as f64 * d, w[0] + (m + 1) as f64 * d))
                    .collect()
            })
            .collect();
        let mut centres = Vec::with_capacity(c.iter().product());
        for i in 0..c[0] {
            let rc = 0.5 * (r[i] + r[i + 1]);
            for j in 0..c[1] {
                let ac = 0.5 * (a[j] + a[j + 1]);
                for k in 0..c[2] {
                    centres.push([rc * ac.cos(), rc * ac.sin(), 0.5 * (z[k] + z[k + 1])]);
                }
            }
        }
        let s_max = 2.0 * (2.0 * r[c[0]] + (z[c[2]] - z[0])) + 1e-300;
        Self {
            v,
            mass,
            opts: *opts,
            c,
            cols: Columns { nz: c[2], data },
            wedges,
            centres,
            total_mass: v.tissue_mass(),
            s_max,
        }
    }

    /// The `(ρ, α)` columns whose cells may overlap `rect` (`[x0, y0, x1,
    /// y1]`), with `f(i, j)` called for each.
    fn columns_in(&self, rect: [f64; 4], mut f: impl FnMut(usize, usize)) {
        let [x0, y0, x1, y1] = rect;
        let (r, a) = (&self.v.lines[0], &self.v.lines[1]);
        let cx = 0.0f64.clamp(x0, x1);
        let cy = 0.0f64.clamp(y0, y1);
        let dmin = cx.hypot(cy);
        let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        let dmax = corners.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f64::max);
        let i0 = r.partition_point(|&x| x <= dmin).saturating_sub(1);
        let i1 = r.partition_point(|&x| x < dmax).min(self.c[0]);
        // Angular interval of the rectangle (all angles if it holds the
        // origin).
        let ang = (dmin > 0.0).then(|| {
            let tc = (0.5 * (y0 + y1)).atan2(0.5 * (x0 + x1));
            let w = corners
                .iter()
                .map(|p| {
                    let d = p[1].atan2(p[0]) - tc;
                    (d - core::f64::consts::TAU * (d / core::f64::consts::TAU).round()).abs()
                })
                .fold(0.0, f64::max);
            (tc - w, tc + w)
        });
        for i in i0..i1 {
            for j in 0..self.c[1] {
                let hit = ang.is_none_or(|(lo, hi)| {
                    [-core::f64::consts::TAU, 0.0, core::f64::consts::TAU]
                        .iter()
                        .any(|&s| a[j] <= hi + s && a[j + 1] >= lo + s)
                });
                if hit {
                    f(i, j);
                }
            }
        }
    }

    /// `[mass, power, tissue volume, volume]` in the box `lo..hi`.
    fn integrate(&self, lo: [f64; 3], hi: [f64; 3]) -> [f64; 4] {
        let (zs, nzs) = segments(&self.v.lines[2], lo[2], hi[2]);
        let mut out = [0.0; 4];
        if nzs == 0 {
            return out;
        }
        let rect = [lo[0], lo[1], hi[0], hi[1]];
        let r = &self.v.lines[0];
        self.columns_in(rect, |i, j| {
            let area: f64 = self.wedges[j]
                .iter()
                .map(|&(a0, a1)| rect_in_sector(rect, r[i], r[i + 1], a0, a1))
                .sum();
            if area <= 0.0 {
                return;
            }
            let col = i * self.c[1] + j;
            for &(rz, wz) in &zs[..nzs] {
                let s = self.cols.sum(col, rz);
                for q in 0..4 {
                    out[q] += area * wz * s[q];
                }
            }
        });
        out
    }

    fn bounds(&self, cell: usize, p: Placement, s: f64) -> ([f64; 3], [f64; 3]) {
        let x = self.centres[cell];
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for a in 0..3 {
            (lo[a], hi[a]) = match p {
                Placement::Face { axis, up } if axis == a => {
                    if up {
                        (x[a], x[a] + s)
                    } else {
                        (x[a] - s, x[a])
                    }
                }
                _ => (x[a] - 0.5 * s, x[a] + 0.5 * s),
            };
        }
        (lo, hi)
    }

    fn cube(&self, cell: usize, p: Placement) -> Option<Cube> {
        let m = self.mass;
        if self.total_mass < m * (1.0 - 1e-12) {
            return None;
        }
        let eval = |s: f64| {
            let (lo, hi) = self.bounds(cell, p, s);
            self.integrate(lo, hi)
        };
        let f = |s: f64| eval(s)[0] - m;
        let rho = self.v.density[cell].max(1e-300);
        let s_max = self.s_max;
        let mut hi = (m / rho).cbrt().min(s_max);
        let mut f_hi = f(hi);
        while f_hi < 0.0 {
            if hi >= s_max {
                return None;
            }
            hi = (2.0 * hi).min(s_max);
            f_hi = f(hi);
        }
        let (mut lo, mut f_lo) = (0.0, -m);
        // Illinois regula falsi, as for the Cartesian cubes.
        let mut side = 0;
        let mut s = hi;
        for _ in 0..200 {
            s = (lo * f_hi - hi * f_lo) / (f_hi - f_lo);
            if !(s > lo && s < hi) {
                s = 0.5 * (lo + hi);
            }
            let fs = f(s);
            if fs.abs() <= 1e-12 * m || hi - lo <= 1e-13 * hi {
                break;
            }
            if fs > 0.0 {
                hi = s;
                f_hi = fs;
                if side == 1 {
                    f_lo *= 0.5;
                }
                side = 1;
            } else {
                lo = s;
                f_lo = fs;
                if side == -1 {
                    f_hi *= 0.5;
                }
                side = -1;
            }
        }
        let [mass, power, tissue, volume] = eval(s);
        if mass <= 0.0 {
            return None;
        }
        Some(Cube {
            side: s,
            sar: power / mass,
            air: (1.0 - tissue / (s * s * s)).max(0.0),
            mass,
            volume,
        })
    }

    fn par_map<T: Send + Clone + Default>(
        &self,
        n: usize,
        f: impl Fn(usize) -> T + Sync,
    ) -> Vec<T> {
        let t = match self.opts.threads {
            0 => std::thread::available_parallelism().map_or(1, usize::from),
            n => n,
        }
        .clamp(1, n.max(1));
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

    fn run(&self) -> SarField {
        let n: usize = self.c.iter().product();
        let tissue = |i: usize| self.v.density[i] > 0.0;
        let centred: Vec<Option<Cube>> = self.par_map(n, |i| {
            if !tissue(i) {
                return None;
            }
            self.cube(i, Placement::Centered)
                .filter(|q| q.air <= self.opts.max_air_fraction)
        });
        let half_max = centred
            .iter()
            .flatten()
            .map(|q| 0.5 * q.side)
            .fold(0.0, f64::max);
        let zc: Vec<f64> = self.v.lines[2]
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect();
        let rest: Vec<(f64, VoxelClass, [f64; 2])> = self.par_map(n, |i| {
            if !tissue(i) {
                return (0.0, VoxelClass::NoTissue, [0.0; 2]);
            }
            if let Some(q) = centred[i] {
                return (q.sar, VoxelClass::Centered, [q.mass, q.volume]);
            }
            let x = self.centres[i];
            let h = half_max;
            let k0 = zc.partition_point(|&y| y < x[2] - h);
            let k1 = zc.partition_point(|&y| y <= x[2] + h);
            let mut best: Option<f64> = None;
            self.columns_in([x[0] - h, x[1] - h, x[0] + h, x[1] + h], |p, q| {
                for r in k0..k1 {
                    let j = (p * self.c[1] + q) * self.c[2] + r;
                    if let Some(Cube { sar, side, .. }) = centred[j] {
                        let y = self.centres[j];
                        let inside = (0..3).all(|a| (x[a] - y[a]).abs() <= 0.5 * side);
                        if inside && best.is_none_or(|b| sar > b) {
                            best = Some(sar);
                        }
                    }
                }
            });
            if let Some(b) = best {
                return (b, VoxelClass::Covered, [0.0; 2]);
            }
            let mut pick: Option<Cube> = None;
            for axis in 0..3 {
                for up in [false, true] {
                    if let Some(q) = self.cube(i, Placement::Face { axis, up }) {
                        let better = match pick {
                            None => true,
                            Some(b) => q.air < b.air || (q.air == b.air && q.sar > b.sar),
                        };
                        if better {
                            pick = Some(q);
                        }
                    }
                }
            }
            match pick {
                Some(q) => (q.sar, VoxelClass::Boundary, [q.mass, q.volume]),
                None => (0.0, VoxelClass::Unresolved, [0.0; 2]),
            }
        });
        let mut values = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(n);
        let mut cube_mass = Vec::with_capacity(n);
        let mut cube_volume = Vec::with_capacity(n);
        for (v, c, [m, vol]) in rest {
            values.push(v);
            classes.push(c);
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
            cube_face: Vec::new(),
        }
    }
}

/// The physical-cube average of cylindrical cells (see the
/// [module docs](self)); `mass` and `opts` are validated by the caller.
pub(super) fn average(v: &SarVolume, mass: f64, opts: &AveragingOptions) -> SarField {
    CylAverager::new(v, mass, opts).run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    #[test]
    fn sector_areas_are_exact() {
        // Whole sector inside a big square.
        let big = [-10.0, -10.0, 10.0, 10.0];
        let a = rect_in_sector(big, 1.0, 2.0, 0.1, 0.6);
        assert!((a - 0.5 * 0.5 * (4.0 - 1.0)).abs() < 1e-14, "{a}");
        // Quarter disc of radius 1 inside the unit square at the origin.
        let a = rect_in_sector([0.0, 0.0, 1.0, 1.0], 0.0, 1.0, 0.0, FRAC_PI_2);
        assert!((a - PI / 4.0).abs() < 1e-14, "{a}");
        // Square of side 1 at the origin, sector radius √2 covers it.
        let a = rect_in_sector(
            [0.0, 0.0, 1.0, 1.0],
            0.0,
            2f64.sqrt() + 1e-12,
            0.0,
            FRAC_PI_2,
        );
        assert!((a - 1.0).abs() < 1e-10, "{a}");
        // Annulus split into cells sums to the square's area inside it.
        let rect = [0.3, -0.7, 1.4, 0.9];
        let r = [0.0, 0.5, 1.0, 1.5, 2.0];
        let na = 7;
        let mut sum = 0.0;
        for i in 0..4 {
            for j in 0..na {
                let a0 = j as f64 * TAU / na as f64;
                let a1 = (j + 1) as f64 * TAU / na as f64;
                let n = ((a1 - a0) / FRAC_PI_2).ceil() as usize;
                for m in 0..n {
                    let d = (a1 - a0) / n as f64;
                    sum += rect_in_sector(
                        rect,
                        r[i],
                        r[i + 1],
                        a0 + m as f64 * d,
                        a0 + (m + 1) as f64 * d,
                    );
                }
            }
        }
        assert!((sum - 1.1 * 1.6).abs() < 1e-12, "{sum}");
    }

    /// Cylinder `ρ ≤ 20 mm`, `0 ≤ z ≤ 24 mm` with an axis, 2 mm × 16 × 2 mm
    /// cells; `tissue(ρ, z)` decides the tissue cells, `p(x, y, z)` the
    /// power density at the cell centre.
    fn cylinder(tissue: impl Fn(f64, f64) -> bool, p: impl Fn(f64, f64, f64) -> f64) -> SarVolume {
        let r: Vec<f64> = (0..=10).map(|i| 2e-3 * f64::from(i)).collect();
        let a: Vec<f64> = (0..=16).map(|j| TAU * f64::from(j) / 16.0).collect();
        let z: Vec<f64> = (0..=12).map(|k| 2e-3 * f64::from(k)).collect();
        let (mut pw, mut d) = (Vec::new(), Vec::new());
        for i in 0..10 {
            let rc = 0.5 * (r[i] + r[i + 1]);
            for j in 0..16 {
                let ac = 0.5 * (a[j] + a[j + 1]);
                for k in 0..12 {
                    let zc = 0.5 * (z[k] + z[k + 1]);
                    let t = tissue(rc, zc);
                    // At the cell's centroid (the cell mean of a linear p).
                    let (r0, r1) = (r[i], r[i + 1]);
                    let h = 0.5 * (a[j + 1] - a[j]);
                    let rb =
                        2.0 / 3.0 * (r1.powi(3) - r0.powi(3)) / (r1 * r1 - r0 * r0) * h.sin() / h;
                    pw.push(p(rb * ac.cos(), rb * ac.sin(), zc));
                    d.push(if t { 1000.0 } else { 0.0 });
                }
            }
        }
        SarVolume::cylindrical([r, a, z], pw, d).unwrap()
    }

    #[test]
    fn constant_field_averages_to_local_everywhere() {
        // Tissue ρ < 14 mm, 4 < z < 20 mm inside air: centred, covered and
        // face cubes all average the constant.
        let v = cylinder(
            |r, z| r < 14e-3 && (4e-3..20e-3).contains(&z),
            |_, _, _| 3.0,
        );
        let f = v
            .averaged_with(0.5e-3, &AveragingOptions::default())
            .unwrap();
        let mut kinds = [0usize; 6];
        for (i, (&s, &c)) in f.values().iter().zip(f.classes()).enumerate() {
            kinds[c as usize] += 1;
            if v.density()[i] > 0.0 {
                assert!((s - 3e-3).abs() < 1e-14, "{i}: {s} ({c:?})");
                if matches!(c, VoxelClass::Centered | VoxelClass::Boundary) {
                    assert!((f.cube_masses()[i] / 0.5e-3 - 1.0).abs() < 1e-10);
                }
            } else {
                assert_eq!(c, VoxelClass::NoTissue);
            }
        }
        assert!(kinds[VoxelClass::Centered as usize] > 0, "{kinds:?}");
        assert!(kinds[VoxelClass::Covered as usize] > 0, "{kinds:?}");
        // Thread count does not matter.
        let g = v
            .averaged_with(
                0.5e-3,
                &AveragingOptions {
                    threads: 3,
                    ..AveragingOptions::default()
                },
            )
            .unwrap();
        assert_eq!(f, g);
    }

    #[test]
    fn linear_field_averages_to_the_cube_centre() {
        // All tissue; p linear in the Cartesian x and z. A cube inside
        // uniform tissue averages a linear function to its centre value, up
        // to the piecewise-constant p on the cells the cube cuts (cell
        // means, not means over the cut parts: 4.6e-3 here).
        let p = |x: f64, _y: f64, z: f64| 3.0 + 50.0 * x + 20.0 * z;
        let v = cylinder(|_, _| true, p);
        let m = 0.3e-3;
        let f = v.averaged_with(m, &AveragingOptions::default()).unwrap();
        let mut worst = 0.0f64;
        let mut n = 0;
        let c = v.cells();
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    let cell = (i * c[1] + j) * c[2] + k;
                    // Cubes clear of the domain boundary only.
                    let s = (m / 1000.0f64).cbrt();
                    let rc = 2e-3 * i as f64 + 1e-3;
                    let zc = 2e-3 * k as f64 + 1e-3;
                    if f.classes()[cell] != VoxelClass::Centered
                        || rc + s / 2f64.sqrt() > 20e-3
                        || zc < s / 2.0
                        || zc + s / 2.0 > 24e-3
                    {
                        continue;
                    }
                    n += 1;
                    let [r, a, z] = [0, 1, 2].map(|q| {
                        let l = &v.lines[q];
                        let m = [i, j, k][q];
                        0.5 * (l[m] + l[m + 1])
                    });
                    let want = p(r * a.cos(), r * a.sin(), z) / 1000.0;
                    worst = worst.max((f.values()[cell] / want - 1.0).abs());
                }
            }
        }
        assert!(n > 500, "{n}");
        assert!(worst < 5e-3, "{worst}");
    }

    #[test]
    fn z_layers_average_exactly() {
        // p constant per z cell: a cube inside the tissue averages the z
        // profile over its height exactly (side (m/ρ)^(1/3)).
        let pz = |z: f64| 1.0 + (z * 1e3 / 2.0).floor();
        let v = cylinder(|_, _| true, |_, _, z| pz(z));
        let m = 0.2e-3;
        let s = (m / 1000.0f64).cbrt();
        let f = v.averaged_with(m, &AveragingOptions::default()).unwrap();
        let c = v.cells();
        // Voxel (2, j, 5): ρ = 5 mm, z = 11 mm, cube side 5.85 mm.
        for j in 0..c[1] {
            let cell = (2 * c[1] + j) * c[2] + 5;
            assert_eq!(f.classes()[cell], VoxelClass::Centered);
            let (z0, z1) = (11e-3 - 0.5 * s, 11e-3 + 0.5 * s);
            let mut want = 0.0;
            for k in 0..12 {
                let (a, b) = (2e-3 * k as f64, 2e-3 * (k + 1) as f64);
                want += pz(a + 1e-3) * (b.min(z1) - a.max(z0)).max(0.0);
            }
            want /= s * 1000.0;
            assert!((f.values()[cell] / want - 1.0).abs() < 1e-10, "{j}");
        }
    }
}
