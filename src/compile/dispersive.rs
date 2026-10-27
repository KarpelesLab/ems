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
//! absent: a Drude pole), `τ` = `EpsilonRelaxTime`, all frequencies in Hz
//! (not rad/s) and `τ` a relaxation time (damping `γ = 1/τ`, not
//! `1/(2τ)`); the Debye form is the one documented by csx (`EpsilonDelta`,
//! `EpsilonRelaxTime`). In terms of [`Pole`]: `ω_p = 2π·f_plasma·√ε_e`,
//! `ω_0 = 2π·f_Lor`, `γ = 1/τ`; a relaxation time of 0 means lossless
//! (`γ = 0`, csx docs). The magnetic poles (`Mue…`) map the same way with
//! `μ_e`. Debye: `Δε` and `τ` directly.
//!
//! All of this was confirmed against openEMS v0.37 as a black box
//! (fixtures `tests/fixtures/openems/adv_*`, `tests/openems_adv.rs`): with
//! the same time step, probe signals agree to ~1e-6 for filled lines and
//! slabs of every pole type, electric and magnetic, and for two-pole
//! materials. A 1 % change of `f_plasma` or a factor 1.5 in `τ` moves the
//! signals by 10–40 %, so the parameter meanings are unambiguous.
//!
//! # Which edges get which poles (openEMS semantics)
//!
//! `csx::discretize` only returns the static per-edge values (`ε∞`, `μ∞`,
//! averaged over the quarter / half cells as for any material). The poles
//! are **not** averaged: an E edge gets the full poles of the dispersive
//! property found at its **midpoint** (csx priority query with closed
//! primitive bounds, so a higher-priority material or metal overrides the
//! dispersive one, and an edge exactly on the surface of a box is inside);
//! an H dual edge gets those at its midpoint (the centre of the primary
//! face it pierces). The `√ε` factor of a Lorentz/Drude plasma frequency
//! is the **edge's** effective static `ε_e` (resp. `μ_e`), i.e. the
//! averaged value the engine uses, not the material's own `Epsilon`: on a
//! mesh-aligned interface with vacuum the edge has `ε_e = (1 + ε∞)/2` and
//! the full pole with `ω_p² = (2π·f_plasma)²·ε_e`. Both rules were
//! identified from openEMS output: a slab whose face lies on a mesh line,
//! 0.2 cells off it on either side, or cut through the edge midpoints,
//! gives openEMS's signals to 1e-6 only with them (half-strength poles on
//! the interface edges, ems's earlier volume-fraction average, differ by
//! 5–17 %; the material's `ε∞` instead of the edge's by 2–7 %). Weighting
//! functions of the pole parameters are evaluated at the midpoint.
//!
//! The poles are integrated with the scheme of
//! [`CompileOptions::dispersive_scheme`](super::CompileOptions::dispersive_scheme)
//! (openEMS's explicit leapfrog by default, see
//! [`Scheme`](crate::engine::dispersive::Scheme)). For openEMS files
//! ([`Fdtd::openems_run`](crate::Fdtd::openems_run), Cartesian grids, the
//! leapfrog scheme) the Lorentz and Drude poles run with openEMS's own
//! `f32` arithmetic
//! ([`OpenEmsPole`], bit for bit),
//! and Debye poles whose relaxation time is below three time steps are
//! dropped, as openEMS drops them (design §5.16.12).
//!
//! On cylindrical grids the midpoints are `(ρ, α, z)` points, the
//! candidate edges those in the primitives' `(ρ, z)` bounding box (in the
//! grid's coordinates; the Cartesian box had left out every pole of a
//! primitive that lies at negative x, black box `adv_cyl_drude`
//! variants) and every α line (csx normalises α ranges to `[−π, π)`).
//!
//! Limitations: sampling runs serially.

use csx::properties::{Aniso, DebyeMaterial, LorentzMaterial, weight_vars};
use csx::{BoundingBox, CoordSystem, PropertyKind, PropertyType, TypeMask, Vec3};

use super::NodeMap;
use crate::axis::Axis;

use crate::engine::dispersive::{EdgePole, Field, OpenEmsPole};
use crate::engine::{CoeffRounding, EdgeMaterials, Grid, Pole, e_edge_circuit, h_edge_inductance};

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

/// Poles of `field` along `dir` of a Lorentz material at Cartesian `c`,
/// on an edge whose effective static `ε` (or `μ`) is `inf`; with
/// `raw`, also the material parameters `(ω_p, ω_0, γ)` of each pole kept
/// (unscaled `ω_p = 2π·f_p`, for openEMS's arithmetic).
fn lorentz_poles(
    m: &LorentzMaterial,
    field: Field,
    dir: usize,
    c: Vec3,
    inf: f64,
    out: &mut Vec<Pole>,
    mut raw: Option<&mut Vec<[f64; 3]>>,
) {
    let vars = weight_vars(c);
    let iso = m.material.isotropic;
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
        let (fp, fl, tau) = (
            val(fp, iso, dir, &vars),
            val(fl, iso, dir, &vars),
            val(tau, iso, dir, &vars),
        );
        let pole = lorentz_pole(fp, fl, tau, inf);
        if !pole.is_null() {
            out.push(pole);
            if let Some(r) = raw.as_deref_mut() {
                let gamma = if tau > 0.0 { 1.0 / tau } else { 0.0 };
                r.push([two_pi() * fp, two_pi() * fl, gamma]);
            }
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
    /// The grid's coordinate system (`(ρ, α, z)` points for a cylindrical
    /// grid).
    cs: CoordSystem,
}

impl Sampler<'_> {
    /// Poles of the dispersive property winning at `p` (none for metal,
    /// plain materials or background).
    fn sample(
        &self,
        p: Vec3,
        field: Field,
        dir: usize,
        edge_inf: f64,
        out: &mut Vec<Pole>,
        raw: Option<&mut Vec<[f64; 3]>>,
    ) {
        let Some(h) = self.index.query(p) else {
            return;
        };
        // Weighting functions take Cartesian coordinates.
        let c = match self.cs {
            CoordSystem::Cylindrical => Vec3::new(p[0] * p[1].cos(), p[0] * p[1].sin(), p[2]),
            _ => p,
        };
        match &h.property.kind {
            PropertyKind::LorentzMaterial(m) => {
                lorentz_poles(m, field, dir, c, edge_inf, out, raw);
            }
            PropertyKind::DebyeMaterial(m) if field == Field::Electric => {
                debye_poles(m, dir, c, out);
            }
            _ => {}
        }
    }

    fn line(&self, a: usize, i: usize) -> f64 {
        self.lines[a][i]
    }

    fn mid(&self, a: usize, i: usize) -> f64 {
        0.5 * (self.lines[a][i] + self.lines[a][i + 1])
    }

    /// Poles of the E edge (`field = Electric`) or H dual edge along `d`
    /// at node `n`: those of the property at the edge's midpoint (see the
    /// module docs); the Lorentz/Drude material parameters in `raw`.
    fn edge_poles(
        &self,
        field: Field,
        d: usize,
        n: [usize; 3],
        edge_inf: f64,
        raw: Option<&mut Vec<[f64; 3]>>,
    ) -> Vec<Pole> {
        let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
        let mut p = Vec3::ZERO;
        match field {
            Field::Electric => {
                p[d] = self.mid(d, n[d]);
                p[t1] = self.line(t1, n[t1]);
                p[t2] = self.line(t2, n[t2]);
            }
            Field::Magnetic => {
                p[d] = self.line(d, n[d]);
                p[t1] = self.mid(t1, n[t1]);
                p[t2] = self.mid(t2, n[t2]);
            }
        }
        let mut out = Vec::new();
        self.sample(p, field, d, edge_inf, &mut out, raw);
        out
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
/// Invalid poles are dropped with a warning. On a cylindrical grid the
/// midpoints are `(ρ, α, z)` points.
///
/// With `openems` (the engine grid of an openEMS file), the Lorentz and
/// Drude poles are also returned as [`OpenEmsPole`]s, with the edge's
/// capacitance (inductance) as openEMS's coefficients form it
/// ([`CoeffRounding::OpenEms`]); the first list still holds every pole.
pub(super) fn dispersive_poles(
    model: &csx::ContinuousStructure,
    node_map: &NodeMap,
    materials: &EdgeMaterials,
    openems: Option<&Grid>,
    warnings: &mut Vec<String>,
) -> (Vec<EdgePole>, Vec<OpenEmsPole>) {
    let dispersive =
        TypeMask::of(PropertyType::LorentzMaterial) | TypeMask::of(PropertyType::DebyeMaterial);
    let mut bbox = BoundingBox::EMPTY;
    let mut magnetic = false;
    for p in model.properties_of(dispersive) {
        for q in &p.primitives {
            bbox = bbox.union(&q.bounding_box_in(model.grid.coord_system, model.coord_system));
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
        return (Vec::new(), Vec::new());
    }
    let cs = model.grid.coord_system;
    let s = Sampler {
        index: model.spatial_index(cs, TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal)),
        lines: [0, 1, 2].map(|a| model.grid.lines(a)),
        cs,
    };
    let dims = node_map.nodes();
    // csx normalises a cylindrical bounding box's α range to [−π, π)
    // (design §8.1), which can leave out part of the primitive on a
    // [0, 2π] grid: on cylindrical grids every α line is a candidate (the
    // spatial index decides).
    let cyl = cs == CoordSystem::Cylindrical;
    let ranges: [core::ops::Range<usize>; 3] = core::array::from_fn(|a| {
        if cyl && a == 1 {
            0..dims[1]
        } else {
            node_range(s.lines[a], bbox.min[a], bbox.max[a])
        }
    });
    let mut out = Vec::new();
    let mut oe = Vec::new();
    let mut raw = Vec::new();
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
                        let idx = node_map.engine_index(n);
                        let edge_inf = match field {
                            Field::Electric => materials.eps_r(axis)[idx],
                            Field::Magnetic => materials.mu_r(axis)[idx],
                        };
                        raw.clear();
                        let poles = s.edge_poles(
                            field,
                            d,
                            n,
                            edge_inf,
                            openems.is_some().then_some(&mut raw),
                        );
                        let circuit = openems.filter(|_| !raw.is_empty()).map(|g| match field {
                            Field::Electric => {
                                e_edge_circuit(g, materials, axis, n, CoeffRounding::OpenEms).0
                            }
                            Field::Magnetic => {
                                h_edge_inductance(g, materials, axis, n, CoeffRounding::OpenEms)
                            }
                        });
                        for (q, pole) in poles.into_iter().enumerate() {
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
                            if let Some(circuit) = circuit {
                                let [omega_p, omega_0, gamma] = raw[q];
                                oe.push(OpenEmsPole {
                                    field,
                                    axis,
                                    node: n,
                                    omega_p,
                                    omega_0,
                                    gamma,
                                    circuit,
                                });
                            }
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
    (out, oe)
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

    /// openEMS files: Lorentz/Drude poles get openEMS's arithmetic and
    /// Debye poles with `τ < 3·dt` are dropped; native compiles keep both
    /// as before.
    #[test]
    fn openems_file_poles() {
        use crate::{CompileOptions, Fdtd};
        use csx::properties::{DebyePole, LorentzPole};
        let mut model = csx::ContinuousStructure::new();
        model.grid.delta_unit = 1.0;
        for a in 0..3 {
            model.grid.set_lines(a, (0..=8).map(|q| q as f64)).unwrap();
        }
        let shape = |lo: f64, hi: f64| {
            csx::Primitive::new(csx::primitives::BoxShape::new([lo; 3], [hi; 3]))
        };
        let mut dm = DebyeMaterial::default();
        dm.poles.push(DebyePole::new(3.0, 2e-9));
        let id = model.add_property(csx::Property::debye_material("deb", dm));
        model.add_primitive(id, shape(1.0, 3.0)).unwrap();
        let mut lm = LorentzMaterial::default();
        let mut pole = LorentzPole::zero();
        pole.eps_plasma_frequency.set(2e7);
        pole.eps_relax_time.set(1e-7);
        lm.poles.push(pole);
        let id = model.add_property(csx::Property::lorentz_material("lor", lm));
        model.add_primitive(id, shape(4.0, 7.0)).unwrap();
        for openems in [false, true] {
            let fdtd = Fdtd::new().timestep(1e-9).openems_run(openems).threads(1);
            let c = crate::compile::compile(&model, &fdtd, &CompileOptions::default()).unwrap();
            assert_eq!(c.dispersive_openems.is_empty(), !openems);
            assert!(
                c.dispersive
                    .iter()
                    .all(|e| !openems || matches!(e.pole, Pole::Debye { .. }))
            );
            let (_e, h) = c.build().unwrap();
            assert_eq!(h.debye_poles_ignored > 0, openems);
            assert!(h.dispersive.is_some());
        }
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
