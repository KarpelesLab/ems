//! Interpolation weights between a wire and a rectilinear lattice: the
//! lowest-order Whitney forms (see the [parent module](super)).
//!
//! * **Nodal weights** of a point: the trilinear hat functions `λ_n`.
//! * **Edge weights** of a straight segment `a → b`: for the edge `e`
//!   along axis `d`, `w_e = ∫_seg W_e·dx_d`, where `W_e` is 1 along the
//!   edge's extent in `d` and the linear hats of the edge's node in the
//!   two other axes (so `Σ_e W_e = 1` per component). `w_e/l_e` is the
//!   line integral of the Whitney edge function `W_e·ê/l_e` along the
//!   segment. Because `∇λ_n` is the sum of the Whitney edge functions of
//!   the edges at `n` (with incidence signs), the edge weights satisfy the
//!   discrete charge-conservation identity
//!   `Σ_e D_{n,e}·w_e/l_e = λ_n(b) − λ_n(a)`.
//!
//! The segment is split at the lattice planes; on each piece the
//! integrand is a product of two linear functions of the curve
//! parameter, which the two-point Gauss rule integrates exactly.

use std::collections::BTreeMap;

/// Mesh lines per axis (strictly increasing).
pub(super) type Lines<'a> = [&'a [f64]; 3];

/// Index of the cell of `lines` that contains `v` (clamped to the mesh).
pub(super) fn cell(lines: &[f64], v: f64) -> usize {
    let n = lines.len();
    let i = lines.partition_point(|&x| x <= v);
    i.saturating_sub(1).min(n.saturating_sub(2))
}

/// Cell index and fractional position (0..1) of `v` along one axis.
fn locate(lines: &[f64], v: f64) -> (usize, f64) {
    let c = cell(lines, v);
    let f = ((v - lines[c]) / (lines[c + 1] - lines[c])).clamp(0.0, 1.0);
    (c, f)
}

/// Trilinear (nodal hat) weights of the point `p`: eight `(node, λ)`
/// pairs (some may be 0).
pub(super) fn node_weights(lines: Lines<'_>, p: [f64; 3]) -> [([usize; 3], f64); 8] {
    let l: [(usize, f64); 3] = core::array::from_fn(|d| locate(lines[d], p[d]));
    core::array::from_fn(|k| {
        let mut node = [0; 3];
        let mut w = 1.0;
        for d in 0..3 {
            let up = (k >> d) & 1 == 1;
            node[d] = l[d].0 + usize::from(up);
            w *= if up { l[d].1 } else { 1.0 - l[d].1 };
        }
        (node, w)
    })
}

/// Edge weights `w_e` \[m\] (see the [module docs](self)) of the straight
/// segment `a → b`, added to `out`, keyed by `(axis index, start node)`.
pub(super) fn segment_weights(
    lines: Lines<'_>,
    a: [f64; 3],
    b: [f64; 3],
    out: &mut BTreeMap<(usize, [usize; 3]), f64>,
) {
    let mut taus = vec![0.0, 1.0];
    for d in 0..3 {
        let (lo, hi) = (a[d].min(b[d]), a[d].max(b[d]));
        if hi - lo <= 0.0 {
            continue;
        }
        let ls = lines[d];
        let i0 = ls.partition_point(|&x| x <= lo);
        for &x in &ls[i0..] {
            if x >= hi {
                break;
            }
            taus.push((x - a[d]) / (b[d] - a[d]));
        }
    }
    taus.sort_by(f64::total_cmp);
    taus.dedup();
    let g = 0.5 / 3f64.sqrt();
    let delta: [f64; 3] = core::array::from_fn(|d| b[d] - a[d]);
    for w in taus.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let len = t1 - t0;
        if len <= 0.0 {
            continue;
        }
        let mid = 0.5 * (t0 + t1);
        let at = |t: f64| -> [f64; 3] { core::array::from_fn(|d| a[d] + t * delta[d]) };
        let pm = at(mid);
        // The piece lies in one cell: take it from the midpoint.
        let cells: [usize; 3] = core::array::from_fn(|d| cell(lines[d], pm[d]));
        for t in [mid - g * len, mid + g * len] {
            let p = at(t);
            let f: [f64; 3] = core::array::from_fn(|d| {
                let ls = lines[d];
                let c = cells[d];
                ((p[d] - ls[c]) / (ls[c + 1] - ls[c])).clamp(0.0, 1.0)
            });
            for (d, &dd) in delta.iter().enumerate() {
                if dd == 0.0 {
                    continue;
                }
                let (o1, o2) = ((d + 1) % 3, (d + 2) % 3);
                for k in 0..4 {
                    let (u1, u2) = (k & 1 == 1, k & 2 == 2);
                    let wt = (if u1 { f[o1] } else { 1.0 - f[o1] })
                        * (if u2 { f[o2] } else { 1.0 - f[o2] });
                    if wt == 0.0 {
                        continue;
                    }
                    let mut node = cells;
                    node[o1] += usize::from(u1);
                    node[o2] += usize::from(u2);
                    *out.entry((d, node)).or_insert(0.0) += 0.5 * len * dd * wt;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines() -> [Vec<f64>; 3] {
        [
            vec![0.0, 1.0, 2.5, 3.0, 4.0],
            vec![0.0, 0.7, 1.4, 2.0, 3.3],
            vec![-1.0, 0.0, 1.0, 2.0, 3.0],
        ]
    }

    #[test]
    fn hats_sum_to_one_and_interpolate() {
        let l = lines();
        let lr: Lines<'_> = [&l[0], &l[1], &l[2]];
        let p = [2.7, 1.1, 0.3];
        let w = node_weights(lr, p);
        assert!((w.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-14);
        // Linear functions are reproduced.
        let x: f64 = w.iter().map(|(n, v)| v * l[0][n[0]]).sum();
        assert!((x - p[0]).abs() < 1e-14);
    }

    #[test]
    fn segment_weights_are_whitney() {
        let l = lines();
        let lr: Lines<'_> = [&l[0], &l[1], &l[2]];
        let (a, b) = ([0.3, 0.2, -0.6], [3.4, 2.9, 2.2]);
        let mut w = BTreeMap::new();
        segment_weights(lr, a, b, &mut w);
        // Per component the weights sum to the projection of the segment.
        for d in 0..3 {
            let s: f64 = w.iter().filter(|(k, _)| k.0 == d).map(|(_, v)| v).sum();
            assert!((s - (b[d] - a[d])).abs() < 1e-12, "{d}: {s}");
        }
        // Discrete charge conservation at every node.
        let mut div: BTreeMap<[usize; 3], f64> = BTreeMap::new();
        for (&(d, n), &v) in &w {
            let le = l[d][n[d] + 1] - l[d][n[d]];
            let mut up = n;
            up[d] += 1;
            *div.entry(up).or_insert(0.0) += v / le;
            *div.entry(n).or_insert(0.0) -= v / le;
        }
        let mut want: BTreeMap<[usize; 3], f64> = BTreeMap::new();
        for (n, v) in node_weights(lr, b) {
            *want.entry(n).or_insert(0.0) += v;
        }
        for (n, v) in node_weights(lr, a) {
            *want.entry(n).or_insert(0.0) -= v;
        }
        for (n, v) in div.iter() {
            let e = want.get(n).copied().unwrap_or(0.0);
            assert!((v - e).abs() < 1e-12, "{n:?}: {v} vs {e}");
        }
        for (n, e) in want.iter() {
            assert!((div.get(n).copied().unwrap_or(0.0) - e).abs() < 1e-12);
        }
    }
}
