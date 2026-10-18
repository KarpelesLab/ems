//! Waveguide (mode-matching) probes: `ProbeBox` types `WaveguideVoltage`
//! and `WaveguideCurrent`.
//!
//! The probe's mode function `f = (f_x, f_y, f_z)` is given by the
//! property attributes `ModeFunctionX`, `ModeFunctionY` and
//! `ModeFunctionZ`: expressions over the csx weighting variables
//! ([`csx::properties::WEIGHT_VARS`]: `x, y, z` in **drawing units**,
//! `rho, a, r, t`) and the model parameters; a missing attribute is `0`.
//! The probe box must be flat along its normal (or set `NormDir`); its
//! transverse extent is the integration area `S`. The probe records
//!
//! ```text
//! WaveguideVoltage:  V(t) = w · ∫_S E_t · f dA     (after each E update)
//! WaveguideCurrent:  I(t) = w · ∫_S H_t · f dA     (after each H update)
//! ```
//!
//! with the ProbeBox weight `w` and `f` in units of 1/m for `V` in volts
//! and `I` in amperes (as written by
//! [`RectWaveguidePort`](crate::port::RectWaveguidePort), whose mode
//! functions are normalised to `∫|f|² dA = 1`).
//!
//! # Discretisation
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

use csx::expr::{CompiledExpr, Expr};
use csx::properties::{ProbeBox, ProbeType, WEIGHT_VARS, weight_vars};

use crate::axis::Axis;
use crate::engine::Grid;

/// One probe entry: component, engine node, weight.
pub(crate) type Entry = (Axis, [usize; 3], f64);

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
    if start[n] != stop[n] {
        warnings.push(format!(
            "waveguide probe {name:?}: box is not flat along its normal {normal}; the \
             plane at its start is used"
        ));
    }
    let k = start[n];
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    let nodes = grid.nodes();
    let lines = |a: usize| grid.lines(Axis::from_index(a).expect("axis index in 0..3"));
    let eval = |c: usize, p: [f64; 3]| -> f64 {
        let vars = weight_vars(csx::Vec3::new(
            p[0] / delta_unit,
            p[1] / delta_unit,
            p[2] / delta_unit,
        ));
        funcs[c].eval(&vars)
    };
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
                        let w = eval(ti, p) * clipped(oi, io);
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
                        let base = eval(ti, p) * clipped(ti, it) * d_o / dual;
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
        _ => return None,
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::primitives::BoxShape;
    use csx::{ContinuousStructure, Primitive, Property};

    fn setup(ty: ProbeType, fx: &str, fy: &str) -> (Vec<Entry>, Grid) {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, [0.0, 1.0, 3.0, 4.0]).unwrap();
        m.grid.set_lines(1, [0.0, 2.0, 3.0]).unwrap();
        m.grid.set_lines(2, [0.0, 1.0, 3.0]).unwrap();
        let mut p = Property::probe("p", ProbeBox::new(ty));
        p.set_attribute("ModeFunctionX", fx);
        p.set_attribute("ModeFunctionY", fy);
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
}
