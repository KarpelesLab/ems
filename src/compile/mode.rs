//! Waveguide (mode-matching) probes: `ProbeBox` types `WaveguideVoltage`
//! and `WaveguideCurrent`.
//!
//! The probe's mode function `f = (f_x, f_y, f_z)` is given by the user
//! attributes `ModeFunctionX`, `ModeFunctionY` and `ModeFunctionZ` of the
//! property (the `<Attributes>` child element in the file; openEMS reads
//! them there, not as attributes of `<ProbeBox>`): expressions over the
//! csx weighting variables ([`csx::properties::WEIGHT_VARS`]: `x, y, z`
//! in **drawing units**, `rho, a, r, t`) and the model parameters; a
//! missing attribute is `0`. openEMS's own parser does not know `pi`, so
//! files meant for both codes should write numbers. The probe box must be
//! flat along its normal (or set `NormDir`). Only the tangential
//! components of `f` matter.
//!
//! # openEMS sampling (default)
//!
//! These rules were identified from openEMS v0.37 output (fixtures
//! `adv_wg_*`, design §5.16): with them ems's probe signals agree with
//! openEMS's to 3e-7 (uniform mesh) and 4e-6 (graded mesh). openEMS
//! **normalises the mode function itself**, so only its shape matters and
//! the result is in volts / amperes for any scale of `f`:
//!
//! ```text
//! V = w · Σ_n E_t(n)·f(r_n)·A_n / √(Σ_n |f_t(r_n)|²·A_n)
//! I = w · Σ_c H_t(c)·f(r_c)·A_c / √(Σ_c |f_t(r_c)|²·A_c)
//! ```
//!
//! with the ProbeBox weight `w` and, in the probe plane (never using the
//! outermost mesh lines of the domain):
//!
//! * **Voltage:** the nodes `n` of the box (box faces included), `f` at
//!   the node, `A_n` its dual area, `E_t(n)` the linear interpolation
//!   between the midpoints of the two edges through the node along each
//!   tangential axis.
//! * **Current:** the cells `c` of the box, `f` at the **cell centre**,
//!   `A_c` the cell area, but `H_t(c)` the field at the cell's lower
//!   corner node: the mean of the two dual edges beside the node, each the
//!   mean of the two H planes `k ± ½`. This half-cell offset between `f`
//!   and `H` makes `I` of an exactly matched TE10 mode 0.3 % low at 20
//!   cells across the guide.
//!
//! openEMS also writes a third column, the mode purity
//! `(Σ E_t·f·A)² / (Σ |E_t|²·A · Σ |f_t|²·A)` (the same sums); ems does
//! not record it.
//!
//! # Exact sampling (`emsModeSampling = "exact"`)
//!
//! With the user attribute `emsModeSampling="exact"` (set by
//! [`RectWaveguidePort`](crate::port::RectWaveguidePort); openEMS ignores
//! it) the probe records the plain overlap integrals, with `f` in 1/m:
//!
//! ```text
//! WaveguideVoltage:  V(t) = w · ∫_S E_t · f dA     (after each E update)
//! WaveguideCurrent:  I(t) = w · ∫_S H_t · f dA     (after each H update)
//! ```
//!
//! over the box's transverse extent `S` (the port helper's mode functions
//! are normalised to `∫|f|² dA = 1`).
//!
//! # Discretisation of the exact sampling
//!
//! The integrals become weighted sums of the edge values (`u = E·Δ` on
//! primary edges, `i = H·Δ*` on dual edges), so the probes are plain
//! [`Probe`](crate::engine::Probe) operators:
//!
//! * **Voltage.** Each tangential E edge in the plane (along `t`, at
//!   primary line `k` of the normal axis) carries the strip of width
//!   `w_o` (its dual width across the edge, clipped to `S`); its
//!   contribution `E_t·f_t·Δ_t·w_o = u·f_t·w_o` with `f` sampled at the
//!   edge midpoint.
//! * **Current.** Tangential H lives half a cell off the plane, on the
//!   dual planes `k − ½` and `k + ½`; both are interpolated linearly to
//!   the plane `k` (one of them only on an outer face of the mesh). An H
//!   dual edge along `t` at transverse primary node `t_i`, dual node
//!   `o_{j+½}`, covers `w_t·Δ_o` (dual width along `t` clipped to `S`):
//!   contribution `H_t·f_t·w_t·Δ_o = i·f_t·w_t·Δ_o/Δ*_t`, with `f`
//!   sampled at `(t_i, o_{j+½})` in the plane `k`.
//!
//! In the Yee cell, tangential E and H of a transverse plane sit at the
//! same transverse positions (e.g. `E_y` and `H_x` both at `(x_i,
//! y_{j+½})`), so `V/I` of a single mode is free of transverse sampling
//! error; only the half-cell interpolation of H along the normal adds an
//! `O((βΔ)²)` term.
//!
//! # Cylindrical grids
//!
//! On a cylindrical grid the probe plane must be normal to `z` (a `ρ`–`α`
//! cross section; other normals are ignored with a warning). The mode
//! function components are the `(ρ, α, z)` components (as for csx
//! excitation weights on such grids) and it is evaluated at the Cartesian
//! position of the sample point, so expressions may use `rho` and `a`
//! (or `x`, `y`). The widths above become physical lengths: an `α` width
//! `Δα` at radius `ρ` is `ρ·Δα`. With `u_ρ = E_ρ·Δρ`, `u_α = E_α·ρ_i·Δα`,
//! `i_ρ = H_ρ·Δρ*`, `i_α = H_α·ρ_{i+½}·Δα*`:
//!
//! * `E_ρ` at `(ρ_{i+½}, α_j)`: weight `f_ρ·ρ_{i+½}·Δα*_j`;
//!   `E_α` at `(ρ_i, α_{j+½})`: weight `f_α·Δρ*_i`.
//! * `H_ρ` at `(ρ_i, α_{j+½})`: weight `f_ρ·ρ_i·Δα_j` (its strip
//!   `Δρ*_i × ρ_i·Δα_j` over `l* = Δρ*_i`);
//!   `H_α` at `(ρ_{i+½}, α_j)`: weight `f_α·Δρ_i/Δα*_j`.
//!
//! On a closed azimuth the edges on the `α = 0` and `α = 2π` lines are
//! stored copies of one edge; the clipping gives each copy the half of the
//! strip inside the box, so a box over the full `α` range counts each
//! physical edge once. Edges on the axis (zero length or area) and samples
//! where the mode function is not finite are skipped. Inside an azimuthal
//! multigrid zone (design §5.15.1) the probe reads the stored (fine)
//! representation, which is not the coarse field: a warning is issued.

use csx::expr::{CompiledExpr, Expr};
use csx::properties::{ProbeBox, ProbeType, WEIGHT_VARS, weight_vars};

use crate::axis::Axis;
use crate::engine::Grid;

/// One probe entry: component, engine node, weight.
pub(crate) type Entry = (Axis, [usize; 3], f64);

/// User attribute of a waveguide probe property selecting the sampling:
/// `"exact"` for ems's exact sampling, anything else (or absent) for
/// openEMS's (see the module docs). openEMS ignores the attribute.
pub(crate) const SAMPLING_ATTR: &str = "emsModeSampling";

/// Probe entries of a waveguide probe (`pb.probe_type` is
/// `WaveguideVoltage` or `WaveguideCurrent`) over the snapped box
/// `start`–`stop`; `None` (with a warning) if the probe is unusable.
#[allow(clippy::too_many_arguments)]
pub(super) fn mode_entries(
    model: &csx::ContinuousStructure,
    prop: &csx::Property,
    pb: &ProbeBox,
    grid: &Grid,
    delta_unit: f64,
    start: [usize; 3],
    stop: [usize; 3],
    warnings: &mut Vec<String>,
) -> Option<Vec<Entry>> {
    let name = &prop.name;
    let mut funcs: Vec<CompiledExpr> = Vec::with_capacity(3);
    let mut any = false;
    for c in ["X", "Y", "Z"] {
        let key = format!("ModeFunction{c}");
        let src = prop.attribute(&key).unwrap_or("0");
        any |= prop.attribute(&key).is_some();
        let f = Expr::parse(src).and_then(|e| e.compile(&WEIGHT_VARS, |n| model.parameters.get(n)));
        match f {
            Ok(f) => funcs.push(f),
            Err(e) => {
                warnings.push(format!(
                    "waveguide probe {name:?}: invalid {key} {src:?}: {e}; ignored"
                ));
                return None;
            }
        }
    }
    if !any {
        warnings.push(format!(
            "waveguide probe {name:?}: no ModeFunctionX/Y/Z attributes; it reads 0"
        ));
    }
    let flat: Vec<usize> = (0..3).filter(|&d| start[d] == stop[d]).collect();
    let normal = match pb.norm_dir {
        Some(d) => Axis::from_index(d.index()).expect("axis index in 0..3"),
        None if flat.len() == 1 => Axis::from_index(flat[0]).expect("axis index in 0..3"),
        None => {
            warnings.push(format!(
                "waveguide probe {name:?}: no normal direction (set NormDir or use a box \
                 that is flat along one axis); ignored"
            ));
            return None;
        }
    };
    let n = normal.index();
    let cyl = grid.is_cylindrical();
    if cyl && normal != Axis::Z {
        warnings.push(format!(
            "waveguide probe {name:?}: on a cylindrical grid the probe plane must be \
             normal to z; ignored"
        ));
        return None;
    }
    if start[n] != stop[n] {
        warnings.push(format!(
            "waveguide probe {name:?}: box is not flat along its normal {normal}; the \
             plane at its start is used"
        ));
    }
    let ctx = Ctx {
        pb,
        grid,
        delta_unit,
        funcs: &funcs,
        normal,
        start,
        stop,
    };
    // openEMS's sampling is only identified on Cartesian grids; cylindrical
    // grids always sample exactly.
    Some(if cyl || prop.attribute(SAMPLING_ATTR) == Some("exact") {
        exact_entries(&ctx, name, warnings)
    } else {
        openems_entries(&ctx, name, warnings)
    })
}

/// What both samplings need.
struct Ctx<'a> {
    pb: &'a ProbeBox,
    grid: &'a Grid,
    delta_unit: f64,
    funcs: &'a [CompiledExpr],
    normal: Axis,
    start: [usize; 3],
    stop: [usize; 3],
}

/// ems's exact sampling (see the module docs), selected by the property
/// attribute `emsModeSampling = "exact"`.
fn exact_entries(c: &Ctx<'_>, name: &str, warnings: &mut Vec<String>) -> Vec<Entry> {
    let Ctx {
        pb,
        grid,
        delta_unit,
        funcs,
        normal,
        start,
        stop,
    } = *c;
    let cyl = grid.is_cylindrical();
    let n = normal.index();
    let k = start[n];
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    let nodes = grid.nodes();
    let lines = |a: usize| grid.lines(Axis::from_index(a).expect("axis index in 0..3"));
    let eval = |c: usize, p: [f64; 3]| -> f64 {
        let q = if cyl {
            [p[0] * p[1].cos(), p[0] * p[1].sin(), p[2]]
        } else {
            p
        };
        let vars = weight_vars(csx::Vec3::new(
            q[0] / delta_unit,
            q[1] / delta_unit,
            q[2] / delta_unit,
        ));
        funcs[c].eval(&vars)
    };
    // Metric factor of a width along axis `a` at the point `p`: the radius
    // for `α` on a cylindrical grid, 1 otherwise.
    let scale = |a: usize, p: [f64; 3]| -> f64 { if cyl && a == 1 { p[0] } else { 1.0 } };
    if cyl && (lo[0]..hi[0].max(lo[0] + 1)).any(|i| grid.alpha_factor(i) > 1) {
        warnings.push(format!(
            "waveguide probe {name:?}: inside an azimuthal multigrid zone; it reads the \
             stored fine-grid values, not the coarse field"
        ));
    }
    // Dual width through primary node `i` along `a`, clipped to [lo, hi].
    let clipped = |a: usize, i: usize| -> f64 {
        let l = lines(a);
        let left = if i > lo[a] {
            0.5 * (l[i] - l[i - 1])
        } else {
            0.0
        };
        let right = if i < hi[a] {
            0.5 * (l[i + 1] - l[i])
        } else {
            0.0
        };
        left + right
    };
    let (ta, tb) = normal.others();
    let mut out = Vec::new();
    match pb.probe_type {
        ProbeType::WaveguideVoltage => {
            for (t, o) in [(ta, tb), (tb, ta)] {
                let (ti, oi) = (t.index(), o.index());
                for it in lo[ti]..hi[ti] {
                    for io in lo[oi]..=hi[oi] {
                        let mut node = [0; 3];
                        node[n] = k;
                        node[ti] = it;
                        node[oi] = io;
                        let mut p = [0.0; 3];
                        p[n] = lines(n)[k];
                        p[ti] = 0.5 * (lines(ti)[it] + lines(ti)[it + 1]);
                        p[oi] = lines(oi)[io];
                        let w = eval(ti, p) * clipped(oi, io) * scale(oi, p);
                        if w != 0.0 && w.is_finite() {
                            out.push((t, node, w));
                        }
                    }
                }
            }
        }
        ProbeType::WaveguideCurrent => {
            // Dual planes k − ½ (node k − 1) and k + ½ (node k) with their
            // linear interpolation weights to the plane k.
            let ln = lines(n);
            let below = (k > 0).then(|| (k - 1, 0.5 * (ln[k] - ln[k - 1])));
            let above = (k + 1 < nodes[n]).then(|| (k, 0.5 * (ln[k + 1] - ln[k])));
            let planes: Vec<(usize, f64)> = match (below, above) {
                (Some((kb, db)), Some((ka, da))) => {
                    vec![(kb, da / (da + db)), (ka, db / (da + db))]
                }
                (Some((kb, _)), None) => vec![(kb, 1.0)],
                (None, Some((ka, _))) => vec![(ka, 1.0)],
                (None, None) => Vec::new(),
            };
            for (t, o) in [(ta, tb), (tb, ta)] {
                let (ti, oi) = (t.index(), o.index());
                let ta_ = Axis::from_index(ti).expect("axis index in 0..3");
                for it in lo[ti]..=hi[ti] {
                    let dual = grid.dual_delta(ta_, it);
                    if dual <= 0.0 {
                        continue;
                    }
                    for io in lo[oi]..hi[oi] {
                        let mut p = [0.0; 3];
                        p[n] = ln[k];
                        p[ti] = lines(ti)[it];
                        p[oi] = 0.5 * (lines(oi)[io] + lines(oi)[io + 1]);
                        let d_o = lines(oi)[io + 1] - lines(oi)[io];
                        let base = eval(ti, p) * clipped(ti, it) * d_o * scale(oi, p) / dual;
                        if base == 0.0 || !base.is_finite() {
                            continue;
                        }
                        for &(kp, wp) in &planes {
                            let mut node = [0; 3];
                            node[n] = kp;
                            node[ti] = it;
                            node[oi] = io;
                            out.push((t, node, base * wp));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// openEMS's sampling (see the module docs).
fn openems_entries(c: &Ctx<'_>, name: &str, warnings: &mut Vec<String>) -> Vec<Entry> {
    let Ctx {
        pb,
        grid,
        delta_unit,
        funcs,
        normal,
        start,
        stop,
    } = *c;
    let n = normal.index();
    let k = start[n];
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    let nodes = grid.nodes();
    let ax = |a: usize| Axis::from_index(a).expect("axis index in 0..3");
    let lines = |a: usize| grid.lines(ax(a));
    let eval = |c: usize, p: [f64; 3]| -> f64 {
        let vars = weight_vars(csx::Vec3::new(
            p[0] / delta_unit,
            p[1] / delta_unit,
            p[2] / delta_unit,
        ));
        funcs[c].eval(&vars)
    };
    let (ta, tb) = normal.others();
    let (a, b) = (ta.index(), tb.index());
    // Sample nodes, without the outermost mesh lines: the box's nodes
    // (voltage), the lower corners of the box's cells (current).
    let current = pb.probe_type == ProbeType::WaveguideCurrent;
    let range = |d: usize| {
        let top = if current {
            hi[d].saturating_sub(1)
        } else {
            hi[d]
        };
        lo[d].max(1)..=top.min(nodes[d].saturating_sub(2))
    };
    let mut samples: Vec<([usize; 2], [f64; 2], f64)> = Vec::new();
    for i in range(a) {
        for j in range(b) {
            let mut p = [0.0; 3];
            p[n] = lines(n)[k];
            let area = if current {
                // Cell centre, primary cell area.
                p[a] = 0.5 * (lines(a)[i] + lines(a)[i + 1]);
                p[b] = 0.5 * (lines(b)[j] + lines(b)[j + 1]);
                grid.delta(ta, i) * grid.delta(tb, j)
            } else {
                p[a] = lines(a)[i];
                p[b] = lines(b)[j];
                grid.dual_delta(ta, i) * grid.dual_delta(tb, j)
            };
            samples.push(([i, j], [eval(a, p), eval(b, p)], area));
        }
    }
    let norm2: f64 = samples
        .iter()
        .map(|(_, f, area)| (f[0] * f[0] + f[1] * f[1]) * area)
        .sum();
    if !(norm2 > 0.0 && norm2.is_finite()) {
        warnings.push(format!(
            "waveguide probe {name:?}: the mode function is zero (or not finite) on the \
             probe plane; it reads 0"
        ));
        return Vec::new();
    }
    let scale = 1.0 / norm2.sqrt();
    let mut out = Vec::new();
    let mut push = |axis: Axis, node: [usize; 3], w: f64| {
        if w != 0.0 && w.is_finite() {
            out.push((axis, node, w));
        }
    };
    // H planes k − ½ (node k − 1) and k + ½ (node k), averaged.
    let planes: Vec<(usize, f64)> = match (k > 0, k + 1 < nodes[n]) {
        (true, true) => vec![(k - 1, 0.5), (k, 0.5)],
        (true, false) => vec![(k - 1, 1.0)],
        (false, true) => vec![(k, 1.0)],
        (false, false) => Vec::new(),
    };
    for &([i, j], f, area) in &samples {
        let w = area * scale;
        for (c, (t, o)) in [(ta, tb), (tb, ta)].into_iter().enumerate() {
            let (ti, oi) = (t.index(), o.index());
            let (it, io) = if c == 0 { (i, j) } else { (j, i) };
            let fc = f[c] * w;
            if fc == 0.0 {
                continue;
            }
            let mut node = [0; 3];
            node[n] = k;
            node[oi] = io;
            if current {
                // H_t at the node: mean of the two dual edges beside it
                // along `o`, each the mean of the two H planes.
                for oc in [io - 1, io] {
                    for &(kp, wp) in &planes {
                        node[ti] = it;
                        node[oi] = oc;
                        node[n] = kp;
                        push(t, node, fc * 0.5 * wp / grid.dual_delta(t, it));
                    }
                }
            } else {
                // E_t at the node: linear interpolation between the
                // midpoints of the two edges along `t`.
                let (dl, dh) = (grid.delta(t, it - 1), grid.delta(t, it));
                let (wl, wh) = (dh / (dl + dh), dl / (dl + dh));
                node[ti] = it - 1;
                push(t, node, fc * wl / dl);
                node[ti] = it;
                push(t, node, fc * wh / dh);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::primitives::BoxShape;
    use csx::{ContinuousStructure, Primitive, Property};

    fn setup(ty: ProbeType, fx: &str, fy: &str) -> (Vec<Entry>, Grid) {
        setup_with(ty, fx, fy, true)
    }

    fn setup_with(ty: ProbeType, fx: &str, fy: &str, exact: bool) -> (Vec<Entry>, Grid) {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, [0.0, 1.0, 3.0, 4.0]).unwrap();
        m.grid.set_lines(1, [0.0, 2.0, 3.0]).unwrap();
        m.grid.set_lines(2, [0.0, 1.0, 3.0]).unwrap();
        let mut p = Property::probe("p", ProbeBox::new(ty));
        p.set_attribute("ModeFunctionX", fx);
        p.set_attribute("ModeFunctionY", fy);
        if exact {
            p.set_attribute(SAMPLING_ATTR, "exact");
        }
        let id = m.add_property(p);
        m.add_primitive(
            id,
            Primitive::new(BoxShape::new([0., 0., 1.], [4., 3., 1.])),
        )
        .unwrap();
        let grid = Grid::new([0, 1, 2].map(|a| m.grid.lines(a).iter().map(|x| x * 1e-3).collect()))
            .unwrap();
        let prop = &m.properties()[0];
        let csx::PropertyKind::ProbeBox(pb) = &prop.kind else {
            unreachable!()
        };
        let mut w = Vec::new();
        let e = mode_entries(&m, prop, pb, &grid, 1e-3, [0, 0, 1], [3, 2, 1], &mut w).unwrap();
        assert!(w.is_empty(), "{w:?}");
        (e, grid)
    }

    #[test]
    fn voltage_of_uniform_field() {
        // f = (1, 0, 0) [1/m]: Σ weights·u with u = E·Δx for E_x = 1 V/m
        // must be the area 4 mm × 3 mm.
        let (e, g) = setup(ProbeType::WaveguideVoltage, "1", "0");
        let s: f64 = e
            .iter()
            .map(|&(a, n, w)| {
                assert_eq!(a, Axis::X);
                assert_eq!(n[2], 1);
                w * g.delta(a, n[0])
            })
            .sum();
        assert!((s - 12e-6).abs() < 1e-18, "{s}");
    }

    #[test]
    fn current_of_uniform_field() {
        // f = (0, 1, 0): H_y = 1 A/m everywhere, i = H·Δ*; Σ = area. Both
        // dual planes contribute (weights 2/3 and 1/3 for Δz 1 mm / 2 mm).
        let (e, g) = setup(ProbeType::WaveguideCurrent, "0", "1");
        let mut planes = [0.0; 2];
        let s: f64 = e
            .iter()
            .map(|&(a, n, w)| {
                assert_eq!(a, Axis::Y);
                planes[n[2]] += w * g.dual_delta(a, n[1]);
                w * g.dual_delta(a, n[1])
            })
            .sum();
        assert!((s - 12e-6).abs() < 1e-18, "{s}");
        assert!((planes[0] / s - 2.0 / 3.0).abs() < 1e-12, "{planes:?}");
    }

    #[test]
    fn mode_function_positions() {
        // f_x = y (drawing units): E_x edges sit on y lines.
        let (e, _) = setup(ProbeType::WaveguideVoltage, "y", "0");
        for &(_, n, w) in &e {
            let y = [0.0, 2.0, 3.0][n[1]];
            let wo = [1.0, 1.5, 0.5][n[1]] * 1e-3;
            assert!((w - y * wo).abs() < 1e-15);
        }
        assert_eq!(e.len(), 3 * 2); // y = 0 rows have weight 0
    }

    /// A closed annulus `1 ≤ ρ ≤ 3` mm on a cylindrical grid: uniform
    /// `f_ρ = 1` (voltage) and `f_α = 1` (current) integrate a unit field
    /// to the annulus area `π(b² − a²)`, each seam copy counted once.
    #[test]
    fn cylindrical_annulus_area() {
        use core::f64::consts::{PI, TAU};
        let na = 12;
        let mut m = ContinuousStructure::new();
        m.coord_system = csx::CoordSystem::Cylindrical;
        m.grid.coord_system = csx::CoordSystem::Cylindrical;
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, [1.0, 1.5, 2.5, 3.0]).unwrap();
        m.grid
            .set_lines(1, (0..=na).map(|j| j as f64 * TAU / na as f64))
            .unwrap();
        m.grid.set_lines(2, [0.0, 1.0, 3.0]).unwrap();
        let grid = Grid::cylindrical([
            m.grid.lines(0).iter().map(|x| x * 1e-3).collect(),
            m.grid.lines(1).to_vec(),
            m.grid.lines(2).iter().map(|x| x * 1e-3).collect(),
        ])
        .unwrap();
        let area = PI * (3e-3f64.powi(2) - 1e-3f64.powi(2));
        for (ty, fx, fy, axis) in [
            (ProbeType::WaveguideVoltage, "1", "0", Axis::X),
            (ProbeType::WaveguideCurrent, "0", "1", Axis::Y),
        ] {
            let mut p = Property::probe("p", ProbeBox::new(ty));
            p.set_attribute("ModeFunctionX", fx);
            p.set_attribute("ModeFunctionY", fy);
            let pb = match &p.kind {
                csx::PropertyKind::ProbeBox(pb) => pb.clone(),
                _ => unreachable!(),
            };
            let mut w = Vec::new();
            let e = mode_entries(&m, &p, &pb, &grid, 1e-3, [0, 0, 1], [3, na, 1], &mut w).unwrap();
            assert!(w.is_empty(), "{w:?}");
            // Unit field: u_ρ = Δρ, i_α = ρ_{i+½}·Δα* (the stored value).
            let s: f64 = e
                .iter()
                .map(|&(a, n, wt)| {
                    assert_eq!(a, axis);
                    let edge = if ty == ProbeType::WaveguideVoltage {
                        grid.delta(a, n[0])
                    } else {
                        grid.dual_edge_length(a, n)
                    };
                    wt * edge
                })
                .sum();
            assert!((s / area - 1.0).abs() < 1e-12, "{ty:?}: {s} vs {area}");
        }
    }

    #[test]
    fn openems_voltage_normalised_on_interior_nodes() {
        // Interior nodes (x = 1, 3 mm; y = 2 mm), dual areas 1.5 × 1.5 mm²
        // each: V = Σ E·f·A / √(Σ f²·A) = √(4.5e-6) m for E_x = 1 V/m,
        // whatever the scale of f.
        for fx in ["1", "7"] {
            let (e, g) = setup_with(ProbeType::WaveguideVoltage, fx, "0", false);
            let s: f64 = e.iter().map(|&(a, n, w)| w * g.delta(a, n[0])).sum();
            assert!((s - 4.5e-6f64.sqrt()).abs() < 1e-15, "{s}");
        }
    }

    #[test]
    fn cylindrical_needs_z_normal() {
        let mut m = ContinuousStructure::new();
        m.coord_system = csx::CoordSystem::Cylindrical;
        m.grid.coord_system = csx::CoordSystem::Cylindrical;
        let grid =
            Grid::cylindrical([vec![1e-3, 2e-3], vec![0.0, 0.5, 1.0], vec![0.0, 1e-3]]).unwrap();
        let mut p = Property::probe("p", ProbeBox::new(ProbeType::WaveguideVoltage));
        p.set_attribute("ModeFunctionY", "1");
        let pb = match &p.kind {
            csx::PropertyKind::ProbeBox(pb) => pb.clone(),
            _ => unreachable!(),
        };
        let mut w = Vec::new();
        assert!(mode_entries(&m, &p, &pb, &grid, 1e-3, [0, 0, 0], [1, 2, 0], &mut w).is_some());
        assert!(w.is_empty(), "{w:?}");
        let mut w = Vec::new();
        assert!(mode_entries(&m, &p, &pb, &grid, 1e-3, [0, 1, 0], [1, 1, 1], &mut w).is_none());
        assert!(!w.is_empty());
    }

    #[test]
    fn openems_current_uses_cell_areas() {
        // Nodes (x = 1, 3 mm; y = 2 mm) with their primary cells 2 × 1 and
        // 1 × 1 mm²: I = √(3e-6) m for H_y = 1 A/m.
        let (e, g) = setup_with(ProbeType::WaveguideCurrent, "0", "1", false);
        let s: f64 = e.iter().map(|&(a, n, w)| w * g.dual_delta(a, n[1])).sum();
        assert!((s - 3e-6f64.sqrt()).abs() < 1e-15, "{s}");
    }
}
