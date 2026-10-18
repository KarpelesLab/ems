//! csx `LorentzMaterial` / `DebyeMaterial` → engine dispersive poles
//! ([`crate::engine::dispersive`]).
//!
//! # Parameter conventions
//!
//! csx stores the openEMS file attributes unchanged; their meaning follows
//! the openEMS user documentation (wiki page "Dispersive Material
//! Property"), in the `e^{jωt}` convention:
//!
//! ```text
//! LorentzMaterial: ε(f) = ε0·ε∞·(1 − Σ_p f_plasma,p² / (f² − f_Lor,p² − j·f/(2π·τ_p)))
//! DebyeMaterial:   ε(f) = ε0·(ε∞ + Σ_p Δε_p / (1 + j·2πf·τ_p))
//! ```
//!
//! with `ε∞` the material's `Epsilon`, `f_plasma` =
//! `EpsilonPlasmaFrequency`, `f_Lor` = `EpsilonLorPoleFrequency` (0 or
//! absent: a Drude pole), `τ` = `EpsilonRelaxTime`, all frequencies in Hz;
//! the Debye form is the one documented by csx (`EpsilonDelta`,
//! `EpsilonRelaxTime`). In terms of [`Pole`]: `ω_p = 2π·f_plasma·√ε∞`
//! (note the `ε∞` factor of the openEMS Lorentz/Drude form, which csx's
//! own field docs omit), `ω_0 = 2π·f_Lor`, `γ = 1/τ`; a relaxation time of
//! 0 means lossless (`γ = 0`, csx docs). The magnetic poles (`Mue…`) map
//! the same way with `μ∞`. Debye: `Δε` and `τ` directly.
//!
//! # Which edges get which poles
//!
//! `csx::discretize` only returns the static per-edge values (`ε∞`, `μ∞`).
//! The poles are sampled here the same way csx samples `ε`: every E edge
//! near a dispersive primitive is averaged over the (up to four) quarter
//! cells of its dual face, each sampled at the centres of an `N×N×N`
//! sub-grid ([`CompileOptions::subsampling`](super::CompileOptions::subsampling)),
//! with the csx priority query (so a higher-priority material or metal
//! overrides the dispersive one). A sample's poles enter with its volume
//! weight divided by the total weight of the non-metal samples, exactly the
//! weights of the static `ε` average: an edge on a mesh-aligned interface
//! gets half the susceptibility, like it gets the mean `ε`. Identical
//! poles are merged per edge. Magnetic poles are averaged arithmetically
//! over the two half cells of the dual edge (exact for homogeneous dual
//! edges; the static `μ` uses the harmonic mean). Weighting functions of
//! the pole parameters are evaluated at each sample point.
//!
//! Limitations: sampling runs serially; the `ε∞` factor of a Lorentz pole
//! is the material's own `ε` at the sample point (not the edge average).

use csx::properties::{Aniso, DebyeMaterial, LorentzMaterial, weight_vars};
use csx::{BoundingBox, CoordSystem, PropertyKind, PropertyType, TypeMask, Vec3};

use super::NodeMap;
use crate::axis::Axis;
use crate::engine::Pole;
use crate::engine::dispersive::{EdgePole, Field};

/// Value of `a` along `dir` at `vars` (isotropic: axis 0).
fn val(a: &Aniso, iso: bool, dir: usize, vars: &[f64; 7]) -> f64 {
    let d = if iso { 0 } else { dir };
    if a.weight[d].is_unity() {
        a.base(d)
    } else {
        a.at_vars(d, vars)
    }
}

fn two_pi() -> f64 {
    2.0 * core::f64::consts::PI
}

/// One Lorentz/Drude pole from openEMS parameters (Hz, s) with the `ε∞`
/// (or `μ∞`) factor.
fn lorentz_pole(f_p: f64, f_l: f64, tau: f64, inf: f64) -> Pole {
    let gamma = if tau > 0.0 { 1.0 / tau } else { 0.0 };
    let omega_p = two_pi() * f_p * inf.max(0.0).sqrt();
    if f_l == 0.0 {
        Pole::drude(omega_p, gamma)
    } else {
        Pole::lorentz(omega_p, two_pi() * f_l, gamma)
    }
}

/// Poles of `field` along `dir` of a Lorentz material at Cartesian `c`.
fn lorentz_poles(m: &LorentzMaterial, field: Field, dir: usize, c: Vec3, out: &mut Vec<Pole>) {
    let vars = weight_vars(c);
    let iso = m.material.isotropic;
    let inf = match field {
        Field::Electric => val(&m.material.epsilon, iso, dir, &vars),
        Field::Magnetic => val(&m.material.mue, iso, dir, &vars),
    };
    for p in &m.poles {
        let (fp, fl, tau) = match field {
            Field::Electric => (
                &p.eps_plasma_frequency,
                &p.eps_lorentz_pole_frequency,
                &p.eps_relax_time,
            ),
            Field::Magnetic => (
                &p.mue_plasma_frequency,
                &p.mue_lorentz_pole_frequency,
                &p.mue_relax_time,
            ),
        };
        let pole = lorentz_pole(
            val(fp, iso, dir, &vars),
            val(fl, iso, dir, &vars),
            val(tau, iso, dir, &vars),
            inf,
        );
        if !pole.is_null() {
            out.push(pole);
        }
    }
}

/// Debye poles along `dir` at Cartesian `c`.
fn debye_poles(m: &DebyeMaterial, dir: usize, c: Vec3, out: &mut Vec<Pole>) {
    let vars = weight_vars(c);
    let iso = m.material.isotropic;
    for p in &m.poles {
        let pole = Pole::debye(
            val(&p.eps_delta, iso, dir, &vars),
            val(&p.eps_relax_time, iso, dir, &vars),
        );
        if !pole.is_null() {
            out.push(pole);
        }
    }
}

/// Sampler over one model.
struct Sampler<'a> {
    index: csx::structure::SpatialIndex<'a>,
    lines: [&'a [f64]; 3],
    n_sub: usize,
}

/// Result of one point sample.
enum Sample {
    Metal,
    Plain,
    Poles,
}

impl Sampler<'_> {
    fn sample(&self, p: Vec3, field: Field, dir: usize, out: &mut Vec<Pole>) -> Sample {
        let Some(h) = self.index.query(p) else {
            return Sample::Plain;
        };
        match &h.property.kind {
            PropertyKind::Metal => Sample::Metal,
            PropertyKind::LorentzMaterial(m) => {
                lorentz_poles(m, field, dir, p, out);
                Sample::Poles
            }
            PropertyKind::DebyeMaterial(m) if field == Field::Electric => {
                debye_poles(m, dir, p, out);
                Sample::Poles
            }
            _ => Sample::Plain,
        }
    }

    /// Integrates the box `[lo, hi]`: adds `(pole, weight)` pairs to
    /// `acc` and returns the non-metal weight.
    fn integrate(
        &self,
        lo: Vec3,
        hi: Vec3,
        field: Field,
        dir: usize,
        acc: &mut Vec<(Pole, f64)>,
    ) -> f64 {
        let n = self.n_sub;
        let step = (hi - lo) / n as f64;
        let w = (0..3).map(|d| step[d].abs()).product::<f64>();
        let mut wsum = 0.0;
        let mut poles = Vec::new();
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let p = Vec3::new(
                        lo[0] + step[0] * (a as f64 + 0.5),
                        lo[1] + step[1] * (b as f64 + 0.5),
                        lo[2] + step[2] * (c as f64 + 0.5),
                    );
                    poles.clear();
                    match self.sample(p, field, dir, &mut poles) {
                        Sample::Metal => continue,
                        Sample::Plain => {}
                        Sample::Poles => {
                            for &pole in &poles {
                                match acc.iter_mut().find(|(q, _)| *q == pole) {
                                    Some((_, x)) => *x += w,
                                    None => acc.push((pole, w)),
                                }
                            }
                        }
                    }
                    wsum += w;
                }
            }
        }
        wsum
    }

    fn line(&self, a: usize, i: usize) -> f64 {
        self.lines[a][i]
    }

    /// Signed half cell next to line `i` on side `s` along `a`.
    fn half(&self, a: usize, i: usize, up: bool) -> Option<f64> {
        let l = self.lines[a];
        if up {
            (i + 1 < l.len()).then(|| 0.5 * (l[i + 1] - l[i]))
        } else {
            (i >= 1).then(|| -0.5 * (l[i] - l[i - 1]))
        }
    }

    /// Effective poles of the E edge (`field = Electric`) or H dual edge
    /// along `d` at node `n`.
    fn edge_poles(&self, field: Field, d: usize, n: [usize; 3]) -> Vec<(Pole, f64)> {
        let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
        let mut acc = Vec::new();
        let mut wsum = 0.0;
        let (mut lo, mut hi) = (Vec3::ZERO, Vec3::ZERO);
        match field {
            Field::Electric => {
                lo[d] = self.line(d, n[d]);
                hi[d] = self.line(d, n[d] + 1);
                for u1 in [false, true] {
                    let Some(h1) = self.half(t1, n[t1], u1) else {
                        continue;
                    };
                    for u2 in [false, true] {
                        let Some(h2) = self.half(t2, n[t2], u2) else {
                            continue;
                        };
                        lo[t1] = self.line(t1, n[t1]);
                        hi[t1] = lo[t1] + h1;
                        lo[t2] = self.line(t2, n[t2]);
                        hi[t2] = lo[t2] + h2;
                        wsum += self.integrate(lo, hi, field, d, &mut acc);
                    }
                }
            }
            Field::Magnetic => {
                for t in [t1, t2] {
                    lo[t] = self.line(t, n[t]);
                    hi[t] = self.line(t, n[t] + 1);
                }
                for up in [false, true] {
                    let Some(h) = self.half(d, n[d], up) else {
                        continue;
                    };
                    lo[d] = self.line(d, n[d]);
                    hi[d] = lo[d] + h;
                    wsum += self.integrate(lo, hi, field, d, &mut acc);
                }
            }
        }
        if wsum > 0.0 {
            for (_, w) in &mut acc {
                *w /= wsum;
            }
            acc
        } else {
            Vec::new()
        }
    }
}

/// Node range on one axis whose edges / cells can touch `[lo, hi]`.
fn node_range(lines: &[f64], lo: f64, hi: f64) -> core::ops::Range<usize> {
    let n = lines.len();
    let first = lines.iter().position(|&x| x >= lo).unwrap_or(n);
    let start = first.saturating_sub(1);
    let end = lines
        .iter()
        .rposition(|&x| x <= hi)
        .map_or(0, |l| (l + 2).min(n));
    if start < end { start..end } else { 0..0 }
}

/// The dispersive poles of `model` per engine edge (see the module docs).
/// Invalid poles are dropped with a warning. The model's grid must be
/// Cartesian.
pub(super) fn dispersive_poles(
    model: &csx::ContinuousStructure,
    node_map: &NodeMap,
    subsampling: usize,
    warnings: &mut Vec<String>,
) -> Vec<EdgePole> {
    let dispersive =
        TypeMask::of(PropertyType::LorentzMaterial) | TypeMask::of(PropertyType::DebyeMaterial);
    let mut bbox = BoundingBox::EMPTY;
    let mut magnetic = false;
    for p in model.properties_of(dispersive) {
        for q in &p.primitives {
            bbox = bbox.union(&q.bounding_box(model.coord_system));
        }
        if let PropertyKind::LorentzMaterial(m) = &p.kind {
            magnetic |= m.poles.iter().any(|q| {
                q.mue_plasma_frequency
                    .value
                    .iter()
                    .any(|v| v.value() != 0.0)
            });
        }
    }
    if bbox.is_empty() || !(bbox.min.is_finite() && bbox.max.is_finite()) {
        return Vec::new();
    }
    let s = Sampler {
        index: model.index(
            CoordSystem::Cartesian,
            TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal),
        ),
        lines: [0, 1, 2].map(|a| model.grid.lines(a)),
        n_sub: subsampling.max(1),
    };
    let dims = node_map.nodes();
    let ranges: [core::ops::Range<usize>; 3] =
        core::array::from_fn(|a| node_range(s.lines[a], bbox.min[a], bbox.max[a]));
    let mut out = Vec::new();
    let mut bad = 0usize;
    let fields: &[Field] = if magnetic {
        &[Field::Electric, Field::Magnetic]
    } else {
        &[Field::Electric]
    };
    for &field in fields {
        for axis in Axis::ALL {
            let d = axis.index();
            let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
            for i in ranges[0].clone() {
                for j in ranges[1].clone() {
                    for k in ranges[2].clone() {
                        let n = [i, j, k];
                        let exists = match field {
                            Field::Electric => n[d] + 1 < dims[d],
                            Field::Magnetic => n[t1] + 1 < dims[t1] && n[t2] + 1 < dims[t2],
                        };
                        if !exists {
                            continue;
                        }
                        for (pole, w) in s.edge_poles(field, d, n) {
                            let pole = pole.scaled(w);
                            if pole.validate().is_err() {
                                bad += 1;
                                continue;
                            }
                            out.push(EdgePole {
                                field,
                                axis,
                                node: n,
                                pole,
                            });
                        }
                    }
                }
            }
        }
    }
    if bad > 0 {
        warnings.push(format!(
            "dispersive materials: {bad} edge pole(s) with negative or non-finite parameters ignored"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ranges() {
        let l = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(node_range(&l, 1.5, 2.5), 1..4);
        assert_eq!(node_range(&l, 1.0, 2.0), 0..4);
        assert_eq!(node_range(&l, -5.0, 10.0), 0..5);
        assert_eq!(node_range(&l, 5.0, 6.0), 4..5);
        assert_eq!(node_range(&l, -6.0, -5.0), 0..0);
    }

    #[test]
    fn openems_lorentz_conventions() {
        // ε∞ = 4: ω_p = 2π·f_p·2.
        let Pole::Drude { omega_p, gamma } = lorentz_pole(1e9, 0.0, 1e-9, 4.0) else {
            panic!("Drude pole expected");
        };
        assert!((omega_p / (two_pi() * 2e9) - 1.0).abs() < 1e-15);
        assert!((gamma / 1e9 - 1.0).abs() < 1e-15);
        let p = lorentz_pole(1e9, 3e9, 0.0, 1.0);
        assert_eq!(p, Pole::lorentz(two_pi() * 1e9, two_pi() * 3e9, 0.0));
        // ε(f) = ε∞(1 − f_p²/(f² − f_L² − j f/(2πτ))) at f = 2 GHz.
        let (fp, fl, tau, inf) = (5e9, 10e9, 5e-9, 2.0);
        let f = 2e9;
        let j = crate::Complex64::I;
        let want = inf * (1.0 - fp * fp / (f * f - fl * fl - j * (f / (two_pi() * tau))));
        let got = inf + lorentz_pole(fp, fl, tau, inf).susceptibility(two_pi() * f);
        assert!((got - want).norm() < 1e-12 * want.norm(), "{got} {want}");
    }
}
