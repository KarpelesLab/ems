//! Homogeneous background media, and their detection on the surface of a
//! node box.
//!
//! The plane-wave source ([`PlaneWave`](super::PlaneWave)) and the NF2FF
//! transform ([`crate::nf2ff`]) assume that the surface of their box lies
//! in one homogeneous, isotropic medium: the incident wave and the free-space
//! Green's function are those of that medium. [`Medium::of_box_surface`]
//! finds it from the engine's update coefficients; the compile step does
//! the same from the effective edge materials (exact `f64` values).
//!
//! # Which edges are sampled
//!
//! For every sampled box face (normal `a`, mesh plane `p`), the edges whose
//! material comes from the cell layers on either side of the face: the E
//! edges tangential to the face, the H dual edges crossing it, and the
//! tangential H dual edges half a cell before and after it. Together the
//! six faces see all three components of `ε`, `κ`, `μ` and `σ_m`, so the
//! medium must also be isotropic. PEC edges are skipped.

use super::Engine;
use super::coeffs::Coefficients;
use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use crate::axis::Axis;
use crate::constants::{C0, EPS0, MU0, Z0};
use crate::fdtd::Face;
use crate::{Error, Result};

/// A homogeneous, isotropic (possibly lossy) medium.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Medium {
    /// Relative permittivity.
    pub eps_r: f64,
    /// Relative permeability.
    pub mu_r: f64,
    /// Electric conductivity \[S/m\].
    pub kappa: f64,
    /// Magnetic conductivity \[Ω/m\].
    pub sigma_m: f64,
}

impl Default for Medium {
    fn default() -> Self {
        Self::VACUUM
    }
}

impl Medium {
    /// Vacuum.
    pub const VACUUM: Self = Self {
        eps_r: 1.0,
        mu_r: 1.0,
        kappa: 0.0,
        sigma_m: 0.0,
    };

    /// A lossless medium.
    pub const fn lossless(eps_r: f64, mu_r: f64) -> Self {
        Self {
            eps_r,
            mu_r,
            kappa: 0.0,
            sigma_m: 0.0,
        }
    }

    /// Whether both conductivities are zero.
    pub fn is_lossless(&self) -> bool {
        self.kappa == 0.0 && self.sigma_m == 0.0
    }

    /// Speed of light of the lossless part, `c0/√(ε_r μ_r)` \[m/s\].
    pub fn velocity(&self) -> f64 {
        C0 / (self.eps_r * self.mu_r).sqrt()
    }

    /// Wave impedance of the lossless part, `Z0·√(μ_r/ε_r)` \[Ω\].
    pub fn impedance(&self) -> f64 {
        Z0 * (self.mu_r / self.eps_r).sqrt()
    }

    /// Checks that `ε_r`, `μ_r` are positive and finite and the
    /// conductivities non-negative and finite.
    ///
    /// # Errors
    /// [`Error::Invalid`] otherwise.
    pub fn validate(&self) -> Result<()> {
        let pos = |x: f64| x.is_finite() && x > 0.0;
        let nonneg = |x: f64| x.is_finite() && x >= 0.0;
        if pos(self.eps_r) && pos(self.mu_r) && nonneg(self.kappa) && nonneg(self.sigma_m) {
            Ok(())
        } else {
            Err(Error::Invalid(format!("invalid medium {self:?}")))
        }
    }

    /// The medium on the surface of the node box `lo..=hi` of `engine`'s
    /// grid, reconstructed from the update coefficients (to `f32`
    /// precision), sampling the faces that do not lie on the domain
    /// boundary (see the [module docs](self#which-edges-are-sampled)).
    /// `Ok(None)` if no edge was sampled (all PEC, or every face on the
    /// boundary).
    ///
    /// # Errors
    /// [`Error::Invalid`] if the box does not fit the grid or the surface
    /// is not homogeneous and isotropic (relative tolerance 1e-4). On a
    /// cylindrical grid the coefficients are read with the physical edge
    /// lengths and areas.
    pub fn of_box_surface(engine: &Engine, lo: [usize; 3], hi: [usize; 3]) -> Result<Option<Self>> {
        let grid = engine.grid();
        let n = grid.nodes();
        if (0..3).any(|a| lo[a] > hi[a] || hi[a] >= n[a]) {
            return Err(Error::Invalid(format!(
                "box {lo:?}..={hi:?} does not fit the {n:?} nodes"
            )));
        }
        let faces = interior_faces(grid, lo, hi);
        let co = engine.coefficients();
        let media = CoefficientMedia {
            grid,
            co: &co,
            dt: engine.dt(),
        };
        surface_medium(grid, &media, lo, hi, faces).map_err(Error::Invalid)
    }
}

/// The faces of `lo..=hi` that do not lie on the outer mesh planes.
pub(crate) fn interior_faces(grid: &Grid, lo: [usize; 3], hi: [usize; 3]) -> [bool; 6] {
    let n = grid.nodes();
    Face::ALL.map(|f| {
        let a = f.axis().index();
        if f.is_max() {
            hi[a] + 1 < n[a]
        } else {
            lo[a] > 0
        }
    })
}

/// Per-edge material lookup for [`surface_medium`].
pub(crate) trait EdgeMedia {
    /// `(ε_r, κ)` of the E edge along `axis` at `node`; `None` for PEC.
    fn e_edge(&self, axis: Axis, node: [usize; 3]) -> Option<(f64, f64)>;
    /// `(μ_r, σ_m)` of the H dual edge along `axis` through `node`.
    fn h_edge(&self, axis: Axis, node: [usize; 3]) -> Option<(f64, f64)>;
}

impl EdgeMedia for EdgeMaterials {
    fn e_edge(&self, axis: Axis, node: [usize; 3]) -> Option<(f64, f64)> {
        let i = index(self.nodes(), node);
        (!self.pec_mask(axis)[i]).then(|| (self.eps_r(axis)[i], self.kappa(axis)[i]))
    }
    fn h_edge(&self, axis: Axis, node: [usize; 3]) -> Option<(f64, f64)> {
        let i = index(self.nodes(), node);
        Some((self.mu_r(axis)[i], self.sigma_m(axis)[i]))
    }
}

fn index(n: [usize; 3], [i, j, k]: [usize; 3]) -> usize {
    (i * n[1] + j) * n[2] + k
}

/// Media reconstructed from the update coefficients:
/// `C = (1 + ca)·dt/(2cb)`, `G = (1 − ca)/cb`, and alike for H, with the
/// physical edge lengths and areas of a cylindrical grid (§5.15).
pub(crate) struct CoefficientMedia<'a> {
    pub grid: &'a Grid,
    pub co: &'a Coefficients,
    pub dt: f64,
}

impl EdgeMedia for CoefficientMedia<'_> {
    fn e_edge(&self, axis: Axis, p: [usize; 3]) -> Option<(f64, f64)> {
        let i = self.grid.index(p);
        let (ca, cb) = (
            f64::from(self.co.ca(axis)[i]),
            f64::from(self.co.cb(axis)[i]),
        );
        if cb == 0.0 {
            return None;
        }
        let (b, c) = axis.others();
        let g = self.grid;
        let geo = if g.is_cylindrical() {
            g.dual_face_area(axis, p) / g.edge_length(axis, p)
        } else {
            g.dual_delta(b, p[b.index()]) * g.dual_delta(c, p[c.index()])
                / g.delta(axis, p[axis.index()])
        };
        let cap = (1.0 + ca) * self.dt / (2.0 * cb);
        let cond = (1.0 - ca) / cb;
        Some((cap / (EPS0 * geo), (cond / geo).max(0.0)))
    }
    fn h_edge(&self, axis: Axis, q: [usize; 3]) -> Option<(f64, f64)> {
        let i = self.grid.index(q);
        let (da, db) = (
            f64::from(self.co.da(axis)[i]),
            f64::from(self.co.db(axis)[i]),
        );
        if db == 0.0 {
            return None;
        }
        let (b, c) = axis.others();
        let g = self.grid;
        let geo = if g.is_cylindrical() {
            g.face_area(axis, q) / g.dual_edge_length(axis, q)
        } else {
            g.delta(b, q[b.index()]) * g.delta(c, q[c.index()])
                / g.dual_delta(axis, q[axis.index()])
        };
        let ind = (1.0 + da) * self.dt / (2.0 * db);
        let res = (1.0 - da) / db;
        Some((ind / (MU0 * geo), (res / geo).max(0.0)))
    }
}

/// One sampled value with where it came from (for the error message).
struct Sample {
    what: &'static str,
    axis: Axis,
    node: [usize; 3],
    value: (f64, f64),
}

/// The sampled E and H edges of the `faces` of `lo..=hi` (see the
/// [module docs](self#which-edges-are-sampled)).
fn surface_samples(
    grid: &Grid,
    media: &impl EdgeMedia,
    lo: [usize; 3],
    hi: [usize; 3],
    faces: [bool; 6],
) -> (Vec<Sample>, Vec<Sample>) {
    let n = grid.nodes();
    let (mut e, mut h): (Vec<Sample>, Vec<Sample>) = (Vec::new(), Vec::new());
    for face in Face::ALL {
        if !faces[face.index()] {
            continue;
        }
        let a = face.axis();
        let ai = a.index();
        let p = if face.is_max() { hi[ai] } else { lo[ai] };
        let (b, c) = a.others();
        for (t, u) in [(b, c), (c, b)] {
            let (ti, ui) = (t.index(), u.index());
            // E edges along t on the face.
            for pt in lo[ti]..hi[ti] {
                for pu in lo[ui]..=hi[ui] {
                    let mut q = [0; 3];
                    (q[ai], q[ti], q[ui]) = (p, pt, pu);
                    if let Some(v) = media.e_edge(t, q) {
                        e.push(Sample {
                            what: "E",
                            axis: t,
                            node: q,
                            value: v,
                        });
                    }
                }
            }
            // Tangential H dual edges along t half a cell on either side.
            for pa in [p.wrapping_sub(1), p] {
                if pa + 1 >= n[ai] {
                    continue;
                }
                for pt in lo[ti]..=hi[ti] {
                    for pu in lo[ui]..hi[ui] {
                        let mut q = [0; 3];
                        (q[ai], q[ti], q[ui]) = (pa, pt, pu);
                        if let Some(v) = media.h_edge(t, q) {
                            h.push(Sample {
                                what: "H",
                                axis: t,
                                node: q,
                                value: v,
                            });
                        }
                    }
                }
            }
        }
        // H dual edges crossing the face.
        for pb in lo[b.index()]..hi[b.index()] {
            for pc in lo[c.index()]..hi[c.index()] {
                let mut q = [0; 3];
                (q[ai], q[b.index()], q[c.index()]) = (p, pb, pc);
                if let Some(v) = media.h_edge(a, q) {
                    h.push(Sample {
                        what: "H",
                        axis: a,
                        node: q,
                        value: v,
                    });
                }
            }
        }
    }
    (e, h)
}

/// Tolerance scales of the conductivities (see [`surface_medium`]).
fn loss_scales(grid: &Grid, eps_r: f64, mu_r: f64) -> (f64, f64) {
    let d_min = Axis::ALL
        .map(|ax| grid.min_delta(ax))
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    (EPS0 * eps_r * C0 / d_min, MU0 * mu_r * C0 / d_min)
}

/// Whether two sampled values agree (relative tolerance 1e-4 on `ε`/`μ`,
/// the conductivity against `scale` when larger).
fn agree((v, l): (f64, f64), (v0, l0): (f64, f64), scale: f64) -> bool {
    let tol = 1e-4;
    let close = |x: f64, y: f64, sc: f64| (x - y).abs() <= tol * sc.max(x.abs());
    close(v, v0, v0) && close(l, l0, scale)
}

/// The axis along which the medium on the `faces` of `lo..=hi` is
/// stratified: every sampled edge has the medium of the other sampled
/// edges of the same kind (E/H, axis) at the same index along it. `None`
/// if no axis qualifies (or the surface is homogeneous). The first
/// qualifying axis of the one with the most distinct levels is returned.
pub(crate) fn layered_axis(
    grid: &Grid,
    media: &impl EdgeMedia,
    lo: [usize; 3],
    hi: [usize; 3],
    faces: [bool; 6],
) -> Option<Axis> {
    let (e, h) = surface_samples(grid, media, lo, hi, faces);
    let (ev, hv) = (e.first()?.value, h.first()?.value);
    let (ks, ss) = loss_scales(grid, ev.0, hv.0);
    let mut best: Option<(usize, Axis)> = None;
    for s in Axis::ALL {
        let mut ok = true;
        let mut levels = 0;
        for (list, scale, is_e) in [(&e, ks, true), (&h, ss, false)] {
            // First value per (axis, level).
            let mut seen: Vec<((Axis, usize), (f64, f64))> = Vec::new();
            for x in list.iter() {
                let key = (x.axis, x.node[s.index()]);
                match seen.iter().find(|(k, _)| *k == key) {
                    Some((_, v0)) => {
                        if !agree(x.value, *v0, scale) {
                            ok = false;
                        }
                    }
                    None => seen.push((key, x.value)),
                }
            }
            if is_e {
                let mut distinct: Vec<(f64, f64)> = Vec::new();
                for (_, v) in &seen {
                    if !distinct.iter().any(|d| agree(*v, *d, scale)) {
                        distinct.push(*v);
                    }
                }
                levels = distinct.len();
            }
        }
        if ok && levels > 1 && best.is_none_or(|(n, _)| levels > n) {
            best = Some((levels, s));
        }
    }
    best.map(|(_, s)| s)
}

/// The common medium of the `faces` of `lo..=hi` (see the
/// [module docs](self#which-edges-are-sampled)), `None` without samples,
/// or a description of the first inhomogeneity.
pub(crate) fn surface_medium(
    grid: &Grid,
    media: &impl EdgeMedia,
    lo: [usize; 3],
    hi: [usize; 3],
    faces: [bool; 6],
) -> core::result::Result<Option<Medium>, String> {
    let (e, h) = surface_samples(grid, media, lo, hi, faces);
    let first = |s: &[Sample]| s.first().map(|x| x.value);
    let (ev, hv) = match (first(&e), first(&h)) {
        (None, None) => return Ok(None),
        (Some(ev), None) => (ev, (1.0, 0.0)),
        (None, Some(hv)) => ((1.0, 0.0), hv),
        (Some(ev), Some(hv)) => (ev, hv),
    };
    let medium = Medium {
        eps_r: ev.0,
        kappa: ev.1,
        mu_r: hv.0,
        sigma_m: hv.1,
    };
    // Relative tolerance on ε, μ; the conductivities are compared relative
    // to `ε0·ε_r·c0/Δ_min` (loss tangent 1 at the frequency whose
    // wavelength is the smallest cell) when that is larger than the value.
    let (k_scale, s_scale) = loss_scales(grid, ev.0, hv.0);
    let check = |s: &[Sample], (v0, l0): (f64, f64), scale: f64, names: [&str; 2]| {
        for x in s {
            let (v, l) = x.value;
            if !agree((v, l), (v0, l0), scale) {
                let r = &s[0];
                return Err(format!(
                    "the medium on the box surface is not homogeneous: {} edge along {} at {:?} \
                     has {} = {v}, {} = {l}, but {} edge along {} at {:?} has {} = {v0}, {} = {l0}",
                    x.what,
                    x.axis,
                    x.node,
                    names[0],
                    names[1],
                    r.what,
                    r.axis,
                    r.node,
                    names[0],
                    names[1]
                ));
            }
        }
        Ok(())
    };
    check(&e, ev, k_scale, ["eps_r", "kappa"])?;
    check(&h, hv, s_scale, ["mu_r", "sigma_m"])?;
    Ok(Some(medium))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CellMaterials, Material};

    #[test]
    fn detects_homogeneous_and_rejects_mixed() {
        let g = Grid::uniform([12; 3], [1e-3; 3]).unwrap();
        let bg = Material::dielectric(4.0).with_kappa(0.05).with_mu_r(1.5);
        let mut m = CellMaterials::new(&g, bg);
        let em = EdgeMaterials::from_cells(&g, &m).unwrap();
        let all = [true; 6];
        let got = surface_medium(&g, &em, [3; 3], [8; 3], all)
            .unwrap()
            .unwrap();
        assert!((got.eps_r - 4.0).abs() < 1e-12 && (got.mu_r - 1.5).abs() < 1e-12);
        assert!((got.kappa - 0.05).abs() < 1e-12 && got.sigma_m == 0.0);
        // Engine reconstruction.
        let e = Engine::builder(g.clone())
            .materials(m.clone())
            .threads(1)
            .build()
            .unwrap();
        let r = Medium::of_box_surface(&e, [3; 3], [8; 3]).unwrap().unwrap();
        assert!((r.eps_r / 4.0 - 1.0).abs() < 1e-6, "{r:?}");
        assert!((r.mu_r / 1.5 - 1.0).abs() < 1e-6, "{r:?}");
        assert!((r.kappa / 0.05 - 1.0).abs() < 1e-3, "{r:?}");
        // A scatterer strictly inside does not matter; one touching the
        // surface (cells 7..9 straddle the face at node 8) does.
        m.fill_box([5..6, 5..6, 5..6], Material::dielectric(9.0));
        let em = EdgeMaterials::from_cells(&g, &m).unwrap();
        assert!(surface_medium(&g, &em, [3; 3], [8; 3], all).is_ok());
        m.fill_box([8..9, 5..6, 5..6], Material::dielectric(9.0));
        let em = EdgeMaterials::from_cells(&g, &m).unwrap();
        let err = surface_medium(&g, &em, [3; 3], [8; 3], all).unwrap_err();
        assert!(err.contains("not homogeneous"), "{err}");
        // ... unless that face is not sampled.
        let mut f = all;
        f[Face::XMax.index()] = false;
        assert!(surface_medium(&g, &em, [3; 3], [8; 3], f).is_ok());
        // Anisotropy is inhomogeneity.
        let mut aniso = Material::VACUUM;
        aniso.eps_r = [1.0, 2.0, 1.0];
        let em = EdgeMaterials::from_cells(&g, &CellMaterials::new(&g, aniso)).unwrap();
        assert!(surface_medium(&g, &em, [3; 3], [8; 3], all).is_err());
    }
}
