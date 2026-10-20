//! In-cell corrections of the oblique thin-wire model: the part of the
//! wire's per-unit-length elastance and inductance that the lattice does
//! not represent (see the [parent module](super#in-cell-corrections)).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, PoisonError};

use super::cyl::CylFrame;
use super::deposit::{Lines, cell, node_weights, segment_weights};
use super::lattice::LatticeGreen;

/// `∫₀^L ds / √(|a + s·t − p|² + r²)` over the straight segment `a → b`
/// (`L = |b − a|`, `t` its direction): the reduced thin-wire kernel.
pub(super) fn line_kernel(p: [f64; 3], a: [f64; 3], b: [f64; 3], r: f64) -> f64 {
    let d: [f64; 3] = core::array::from_fn(|k| b[k] - a[k]);
    let len = norm(d);
    if len == 0.0 {
        return 0.0;
    }
    let w: [f64; 3] = core::array::from_fn(|k| p[k] - a[k]);
    let s = dot(w, d) / len;
    let perp2 = (dot(w, w) - s * s).max(0.0);
    let rho = (perp2 + r * r).sqrt();
    ((len - s) / rho).asinh() + (s / rho).asinh()
}

/// `∫∫ ds ds' / √((s − s')² + r²)` over a straight segment of length `l`
/// with itself.
pub(super) fn self_kernel(l: f64, r: f64) -> f64 {
    2.0 * (l * (l / r).asinh() - (l * l + r * r).sqrt() + r)
}

/// `∫_{a→b} ds ∫_{c→e} ds' / √(|x − x'|² + r²)` (outer integral by
/// Gauss–Legendre on a smoothstep-graded parameter, which absorbs the
/// logarithmic end singularities of touching segments; inner analytic).
pub(super) fn pair_kernel(a: [f64; 3], b: [f64; 3], c: [f64; 3], e: [f64; 3], r: f64) -> f64 {
    let len = norm(core::array::from_fn(|k| b[k] - a[k]));
    let mut sum = 0.0;
    for (x, w) in GAUSS12 {
        // τ ∈ [0, 1], s = 3τ² − 2τ³, ds = 6τ(1 − τ) dτ.
        let tau = 0.5 * (x + 1.0);
        let s = tau * tau * (3.0 - 2.0 * tau);
        let ds = 6.0 * tau * (1.0 - tau) * 0.5 * w;
        let p: [f64; 3] = core::array::from_fn(|k| a[k] + s * (b[k] - a[k]));
        sum += ds * line_kernel(p, c, e, r);
    }
    sum * len
}

/// 12-point Gauss–Legendre nodes and weights on [−1, 1].
const GAUSS12: [(f64, f64); 12] = [
    (-0.981_560_634_246_719_3, 0.047_175_336_386_511_83),
    (-0.904_117_256_370_474_9, 0.106_939_325_995_318_4),
    (-0.769_902_674_194_304_7, 0.160_078_328_543_346_2),
    (-0.587_317_954_286_617_4, 0.203_167_426_723_065_9),
    (-0.367_831_498_998_180_2, 0.233_492_536_538_354_8),
    (-0.125_233_408_511_468_9, 0.249_147_045_813_402_8),
    (0.125_233_408_511_468_9, 0.249_147_045_813_402_8),
    (0.367_831_498_998_180_2, 0.233_492_536_538_354_8),
    (0.587_317_954_286_617_4, 0.203_167_426_723_065_9),
    (0.769_902_674_194_304_7, 0.160_078_328_543_346_2),
    (0.904_117_256_370_474_9, 0.106_939_325_995_318_4),
    (0.981_560_634_246_719_3, 0.047_175_336_386_511_83),
];

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The discretised wire geometry the corrections are computed for.
pub(super) struct Geometry<'a> {
    /// Node positions \[m\].
    pub nodes: &'a [[f64; 3]],
    /// Segments `(start node, end node)`.
    pub segs: &'a [(usize, usize)],
    /// Radius per segment \[m\].
    pub radius: &'a [f64],
    /// Connected component per node.
    pub component: &'a [usize],
}

impl Geometry<'_> {
    fn mid(&self, s: usize) -> [f64; 3] {
        let (a, b) = self.segs[s];
        core::array::from_fn(|k| 0.5 * (self.nodes[a][k] + self.nodes[b][k]))
    }
}

/// A uniform virtual lattice around a cell of the real mesh, with the
/// real cell's spacings, on which the lattice response is evaluated.
struct Virtual {
    lines: [Vec<f64>; 3],
    h: [f64; 3],
}

impl Virtual {
    fn around(real: Lines<'_>, p: [f64; 3], reach: f64) -> Self {
        let c: [usize; 3] = core::array::from_fn(|d| cell(real[d], p[d]));
        let h: [f64; 3] = core::array::from_fn(|d| real[d][c[d] + 1] - real[d][c[d]]);
        let lines = core::array::from_fn(|d| {
            let m = (reach / h[d]).ceil() as i64 + 3;
            let x0 = real[d][c[d]];
            (-m..=m + 1).map(|i| x0 + i as f64 * h[d]).collect()
        });
        Self { lines, h }
    }

    fn lines(&self) -> Lines<'_> {
        [&self.lines[0], &self.lines[1], &self.lines[2]]
    }
}

/// Lattice Green's functions per distinct spacing, shared by all wire
/// networks of the process (a table takes a few milliseconds; at most
/// [`CACHE_SIZE`] are kept).
static CACHE: Mutex<BTreeMap<[u64; 3], Arc<LatticeGreen>>> = Mutex::new(BTreeMap::new());
const CACHE_SIZE: usize = 256;

/// The tables used by one computation.
struct Greens(BTreeMap<[u64; 3], Arc<LatticeGreen>>);

impl Greens {
    fn get(&mut self, h: [f64; 3]) -> Arc<LatticeGreen> {
        let key = h.map(f64::to_bits);
        self.0
            .entry(key)
            .or_insert_with(|| {
                let cached = CACHE
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get(&key)
                    .cloned();
                cached.unwrap_or_else(|| {
                    let g = Arc::new(LatticeGreen::new(h, TABLE));
                    let mut c = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
                    if c.len() >= CACHE_SIZE {
                        c.clear();
                    }
                    c.insert(key, Arc::clone(&g));
                    g
                })
            })
            .clone()
    }
}

/// Offsets (cells per axis) tabulated exactly.
const TABLE: usize = 12;

fn offset(a: [usize; 3], b: [usize; 3]) -> [i64; 3] {
    core::array::from_fn(|d| a[d] as i64 - b[d] as i64)
}

/// Per-node capacitance and per-segment inductance corrections (see the
/// [parent module](super#in-cell-corrections)), in vacuum units: with
/// `k(x)` the potential per unit line charge (times `ε`) that the
/// lattice misses at `x`, node `j` gets `∫ ds/k(s)` \[m\] over its half
/// segments (times `ε`: the capacitance; NaN where `k ≤ 0`), and segment
/// `s` the flux per unit current the lattice misses, `Λ_s` \[m\] (times
/// `μ`: the inductance). `window` is the radius \[m\] of the neighbourhood
/// (in cell sizes of the local cell, times the largest of them). On a
/// cylindrical grid (`cyl`) the virtual lattice is the unrolled local
/// frame of the reference point ([`CylFrame::local`]).
pub(super) fn corrections(
    real: Lines<'_>,
    geo: &Geometry<'_>,
    dual: &[f64],
    window: f64,
    cyl: Option<&CylFrame<'_>>,
) -> (Vec<f64>, Vec<f64>) {
    let mut greens = Greens(BTreeMap::new());
    let nn = geo.nodes.len();
    // Half segments per node: (segment, from node to midpoint).
    let mut halves: Vec<Vec<usize>> = vec![Vec::new(); nn];
    for (s, &(a, b)) in geo.segs.iter().enumerate() {
        halves[a].push(s);
        halves[b].push(s);
    }
    let local_h = |p: [f64; 3]| -> f64 {
        if let Some(f) = cyl {
            return f.cell_size(p).into_iter().fold(0.0, f64::max);
        }
        (0..3)
            .map(|d| {
                let c = cell(real[d], p[d]);
                real[d][c + 1] - real[d][c]
            })
            .fold(0.0, f64::max)
    };
    // The virtual lattice around `p` and the map of points into it.
    let frame = |p: [f64; 3], reach: f64| -> (Virtual, Option<super::cyl::Local>) {
        match cyl {
            Some(f) => {
                let loc = f.local(p, reach);
                (
                    Virtual {
                        lines: loc.lines.clone(),
                        h: loc.h,
                    },
                    Some(loc),
                )
            }
            None => (Virtual::around(real, p, reach), None),
        }
    };
    let to = |loc: &Option<super::cyl::Local>, x: [f64; 3]| match loc {
        Some(l) => l.map(x),
        None => x,
    };
    // Gauss–Legendre, 3 points on [0, 1].
    let g3 = [
        (0.5 - 0.5 * 0.6f64.sqrt(), 5.0 / 18.0),
        (0.5, 8.0 / 18.0),
        (0.5 + 0.5 * 0.6f64.sqrt(), 5.0 / 18.0),
    ];
    let mut cap_len = vec![0.0; nn];
    for j in 0..nn {
        let xj = geo.nodes[j];
        let reach = window * local_h(xj);
        let (virt, loc) = frame(xj, reach);
        let vl = virt.lines();
        let g = greens.get(virt.h);
        let mut charge: BTreeMap<[usize; 3], f64> = BTreeMap::new();
        let mut pieces: Vec<([f64; 3], [f64; 3], f64)> = Vec::new();
        for i in 0..nn {
            if geo.component[i] != geo.component[j] {
                continue;
            }
            let xi = geo.nodes[i];
            if norm(core::array::from_fn(|k| xi[k] - xj[k])) > reach {
                continue;
            }
            for (n, w) in node_weights(vl, to(&loc, xi)) {
                if w != 0.0 {
                    *charge.entry(n).or_insert(0.0) += dual[i] * w;
                }
            }
            for &s in &halves[i] {
                pieces.push((xi, geo.mid(s), geo.radius[s]));
            }
        }
        // k(x): correction potential per unit line charge (×ε) at x.
        let k = |x: [f64; 3]| -> f64 {
            let phi: f64 = pieces
                .iter()
                .map(|&(a, b, r)| line_kernel(x, a, b, r))
                .sum::<f64>()
                / (4.0 * PI);
            let mut v = 0.0;
            for (n, lam) in node_weights(vl, to(&loc, x)) {
                if lam == 0.0 {
                    continue;
                }
                for (m, q) in &charge {
                    v += lam * q * g.get(offset(n, *m));
                }
            }
            phi - v
        };
        // ∫ ds/k(s) over the node's half segments.
        let mut acc = 0.0;
        for &s in &halves[j] {
            let m = geo.mid(s);
            let half = norm(core::array::from_fn(|d| m[d] - xj[d]));
            for (t, w) in g3 {
                let x: [f64; 3] = core::array::from_fn(|d| xj[d] + t * (m[d] - xj[d]));
                let kv = k(x);
                acc += if kv > 0.0 { w * half / kv } else { f64::NAN };
            }
        }
        cap_len[j] = acc;
    }
    let ns = geo.segs.len();
    let mut flux = vec![0.0; ns];
    for (s, fl) in flux.iter_mut().enumerate() {
        let ms = geo.mid(s);
        let reach = window * local_h(ms);
        let (virt, loc) = frame(ms, reach);
        let vl = virt.lines();
        let g = greens.get(virt.h);
        let (a, b) = geo.segs[s];
        let (pa, pb) = (geo.nodes[a], geo.nodes[b]);
        let ts = dir(pa, pb);
        let comp = geo.component[a];
        let mut cur: BTreeMap<(usize, [usize; 3]), f64> = BTreeMap::new();
        let mut lam = 0.0;
        for (s2, &(c, e)) in geo.segs.iter().enumerate() {
            if geo.component[c] != comp {
                continue;
            }
            let m2 = geo.mid(s2);
            if norm(core::array::from_fn(|k| m2[k] - ms[k])) > reach {
                continue;
            }
            let (pc, pe) = (geo.nodes[c], geo.nodes[e]);
            segment_weights(vl, to(&loc, pc), to(&loc, pe), &mut cur);
            let cos = dot(ts, dir(pc, pe));
            if cos == 0.0 {
                continue;
            }
            let r = geo.radius[s2];
            lam += cos
                * if s2 == s {
                    self_kernel(norm(core::array::from_fn(|k| pb[k] - pa[k])), r)
                } else {
                    pair_kernel(pa, pb, pc, pe, r)
                };
        }
        lam /= 4.0 * PI;
        let mut own = BTreeMap::new();
        segment_weights(vl, to(&loc, pa), to(&loc, pb), &mut own);
        // Per component: the current elements of that component.
        let by_comp: [Vec<([usize; 3], f64)>; 3] = core::array::from_fn(|d| {
            cur.iter()
                .filter(|(k, _)| k.0 == d)
                .map(|(k, v)| (k.1, *v))
                .collect()
        });
        let mut lat = 0.0;
        for (&(d, n), &w) in &own {
            for &(m, jv) in &by_comp[d] {
                lat += w * jv * g.get(offset(n, m));
            }
        }
        *fl = lam - lat;
    }
    (cap_len, flux)
}

fn dir(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let d: [f64; 3] = core::array::from_fn(|k| b[k] - a[k]);
    let l = norm(d);
    d.map(|v| v / l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::wire::lattice_radius;

    #[test]
    fn kernels() {
        // Self kernel against the pair kernel of two halves.
        let (l, r) = (1.0, 1e-3);
        let a = [0.0; 3];
        let m = [0.5, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let halves = self_kernel(0.5, r) * 2.0 + 2.0 * pair_kernel(a, m, m, b, r);
        assert!(
            (halves / self_kernel(l, r) - 1.0).abs() < 1e-4,
            "{halves} {}",
            self_kernel(l, r)
        );
        // Line kernel of a long straight wire: ≈ 2·ln(2L/r)... symmetric.
        let k = line_kernel([0.0; 3], [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0], 1e-2);
        assert!((k - 2.0 * (10.0f64 / 1e-2).asinh()).abs() < 1e-9);
    }

    /// Straight wire along z through `(x0, y0)` on a uniform cubic mesh:
    /// the equivalent lattice radius `r = r0·e^(2π·D)` of the interior.
    fn radii(x0: f64, y0: f64, seg: f64) -> (f64, f64) {
        let h = 1.0;
        let ls: Vec<f64> = (-40..=40).map(|i| i as f64 * h).collect();
        let real: Lines<'_> = [&ls, &ls, &ls];
        let n = (40.0 / seg) as usize;
        let nodes: Vec<[f64; 3]> = (0..=n).map(|i| [x0, y0, -20.0 + i as f64 * seg]).collect();
        let segs: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();
        let r0 = 1e-3;
        let radius = vec![r0; n];
        let component = vec![0; n + 1];
        let mut dual = vec![seg; n + 1];
        dual[0] = 0.5 * seg;
        dual[n] = 0.5 * seg;
        let geo = Geometry {
            nodes: &nodes,
            segs: &segs,
            radius: &radius,
            component: &component,
        };
        let (c, l) = corrections(real, &geo, &dual, super::super::WINDOW, None);
        let j = n / 2;
        (
            r0 * (2.0 * PI * seg / c[j]).exp(),
            r0 * (2.0 * PI * l[j] / seg).exp(),
        )
    }

    #[test]
    fn axis_aligned_on_node_is_the_lattice_radius() {
        let want = lattice_radius(1.0, 1.0);
        let (re, rm) = radii(0.0, 0.0, 0.5);
        eprintln!("on node: r_e {re:.5}, r_m {rm:.5}, lattice radius {want:.5}");
        assert!((re / want - 1.0).abs() < 1e-2);
        assert!((rm / want - 1.0).abs() < 1e-2);
        // Cell centre: bilinear spreading over four nodes,
        // ln(r/r_eq) = 2π·(2·¼ + 1/π)/16·4 = 2π·(1/8 + 1/(4π)).
        let want_c = want * (2.0 * PI * (0.125 + 0.25 / PI)).exp();
        let (re, rm) = radii(0.5, 0.5, 0.5);
        eprintln!("cell centre: r_e {re:.5}, r_m {rm:.5}, 2-D estimate {want_c:.5}");
        assert!((re / want_c - 1.0).abs() < 1e-2);
        assert!((rm / want_c - 1.0).abs() < 1e-2);
    }
}
