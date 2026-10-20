//! Yee update kernels in integral form (Yee 1966; Taflove & Hagness,
//! *Computational Electrodynamics*, 3rd ed., ch. 3).
//!
//! The kernels are written per x-plane and per `(i, j)` row with the inner
//! k loop over equal-length slices, so LLVM can drop bounds checks and
//! vectorise it. Every value is computed by the same expression in the same
//! order regardless of the slab partition, which makes results bit-identical
//! for any thread count.
//!
//! Out-of-domain neighbours (the `j − 1`, `k − 1` and `i − 1` neighbours at
//! the low faces) read from a zero plane; high-side neighbours are padding
//! entries that are always zero. See the engine module docs for the layout.

use super::fields::{Block, FieldStore, read, write};
use super::rows::CoeffRows;
use super::run::EnergyMeasure;
use crate::constants::{EPS0, MU0};

/// Rounds subnormal results to zero (`|x| ≲ 2^-126`), keeping normal
/// values (absolute error ≤ 2^-126).
///
/// The leading tail of a numerical wave front in an initially empty grid
/// decays geometrically and runs through the subnormal range, where many
/// CPUs are much slower (a plane of subnormal inputs cost ~2× here). Hardware flush-to-zero needs `unsafe` (MXCSR),
/// so we add and subtract a small normal constant instead: with
/// `C = 2^-102`, `ulp(C) = 2^-125`, so any `|x| < 2^-126` rounds to `C`
/// and the difference is exactly zero. The rounding is IEEE-exact, hence
/// deterministic.
#[inline(always)]
pub(crate) fn flush(x: f32) -> f32 {
    const C: f32 = f32::from_bits(25 << 23); // 2^(25 − 127) = 2^-102
    (x + C) - C
}

/// `h = da·h − db·((a − b) − (c − d))` over equal-length slices.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // index form vectorises reliably
fn h_row(h: &mut [f32], da: &[f32], db: &[f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32]) {
    let n = h.len();
    let (da, db, a, b, c, d) = (&da[..n], &db[..n], &a[..n], &b[..n], &c[..n], &d[..n]);
    for k in 0..n {
        h[k] = flush(da[k] * h[k] - db[k] * ((a[k] - b[k]) - (c[k] - d[k])));
    }
}

/// `(1 + a)·p/b`, or 0 where `b = 0` (PEC and padding edges).
///
/// Written branch-free so the energy loops vectorise; where `b ≠ 0` the
/// value is the same expression as a branching form, bit for bit.
#[inline(always)]
fn weight(a: f32, p: f64, b: f32) -> f64 {
    let w = (1.0 + a as f64) * p / b as f64;
    if b != 0.0 { w } else { 0.0 }
}

/// [`h_row`] that also returns the row's share of the leapfrog energy
/// invariant, `Σ (1 + da)/db · h_old · h_new` (entries with `db = 0`
/// skipped), or with `OE` of openEMS's measure `Σ h_new²` (every entry),
/// and writes the new values only if `COMMIT`.
///
/// `h_new` is computed by exactly the same expression as in [`h_row`], so
/// the committed values are bit-identical to a plain update. The sum uses
/// four interleaved `f64` accumulators (entry `k` into `k & 3`) in a fixed
/// order (deterministic); the main loop handles groups of four entries so
/// the accumulators map onto one vector.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn h_row_energy<const COMMIT: bool, const OE: bool>(
    h: &mut [f32],
    da: &[f32],
    db: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
) -> f64 {
    let n = h.len();
    let (da, db, a, b, c, d) = (&da[..n], &db[..n], &a[..n], &b[..n], &c[..n], &d[..n]);
    let mut acc = [0.0f64; 4];
    let mut one = |k: usize, acc: &mut f64| {
        let old = h[k];
        let new = flush(da[k] * old - db[k] * ((a[k] - b[k]) - (c[k] - d[k])));
        *acc += if OE {
            new as f64 * new as f64
        } else {
            weight(da[k], old as f64 * new as f64, db[k])
        };
        if COMMIT {
            h[k] = new;
        }
    };
    let full = n & !3;
    for k0 in (0..full).step_by(4) {
        for (l, acc) in acc.iter_mut().enumerate() {
            one(k0 + l, acc);
        }
    }
    for k in full..n {
        one(k, &mut acc[k & 3]);
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// One H row, plain ([`h_row`], returns 0) or with energy
/// ([`h_row_energy`]).
#[inline(always)]
fn hrow<const ENERGY: bool, const COMMIT: bool, const OE: bool>(
    h: &mut [f32],
    da: &[f32],
    db: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
) -> f64 {
    if ENERGY {
        h_row_energy::<COMMIT, OE>(h, da, db, a, b, c, d)
    } else {
        h_row(h, da, db, a, b, c, d);
        0.0
    }
}

/// Adds `(1 + ca)/cb · u²` of one row to the plane accumulators,
/// skipping `cb = 0` (PEC and padding edges), or with `OE` plain `u²`
/// (every edge). Entry `k` of the row has
/// in-plane index `r + k` and goes to accumulator `(r + k) & 3`, so a plane
/// fed row by row in order sums in a fixed order.
#[allow(clippy::needless_range_loop)]
fn e_energy_row<const OE: bool>(acc: &mut [f64; 4], r: usize, u: &[f32], ca: &[f32], cb: &[f32]) {
    let n = u.len();
    let (ca, cb) = (&ca[..n], &cb[..n]);
    // `loc[l]` is accumulator `(r + l) & 3`, so entry k adds to `loc[k & 3]`.
    let s = r & 3;
    let mut loc: [f64; 4] = core::array::from_fn(|l| acc[(s + l) & 3]);
    let one = |k: usize, acc: &mut f64| {
        let x = u[k] as f64;
        *acc += if OE {
            x * x
        } else {
            weight(ca[k], x * x, cb[k])
        };
    };
    let full = n & !3;
    for k0 in (0..full).step_by(4) {
        for (l, acc) in loc.iter_mut().enumerate() {
            one(k0 + l, acc);
        }
    }
    for k in full..n {
        one(k, &mut loc[k & 3]);
    }
    for (l, v) in loc.into_iter().enumerate() {
        acc[(s + l) & 3] = v;
    }
}

/// `u = ca·u + cb·((a − b) − (c − d))` over equal-length slices.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn e_row(u: &mut [f32], ca: &[f32], cb: &[f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32]) {
    let n = u.len();
    let (ca, cb, a, b, c, d) = (&ca[..n], &cb[..n], &a[..n], &b[..n], &c[..n], &d[..n]);
    for k in 0..n {
        u[k] = flush(ca[k] * u[k] + cb[k] * ((a[k] - b[k]) - (c[k] - d[k])));
    }
}

fn plane(v: &[f32], local: usize, p: usize) -> &[f32] {
    &v[local * p..(local + 1) * p]
}

/// Faraday update of all H dual edges in the x-planes of slab `s`:
/// `i ← da·i − db·∮u` around the primary face each dual edge pierces.
pub(crate) fn update_h_slab(store: &FieldStore, co: &CoeffRows<'_>, s: usize) {
    h_slab::<false, true, false>(store, co, s, &mut |_, _| {});
}

/// [`update_h_slab`] that also evaluates, for every x-plane `gi` of the
/// slab, the plane's share of the leapfrog energy invariant in units of
/// `dt/4` (see [`Engine::energy`](super::Engine::energy)):
///
/// ```text
/// Σ (1 + ca)/cb · (u^n)²  +  Σ (1 + da)/db · i^{n−½} · i^{n+½}
/// ```
///
/// and passes it to `sink(gi, value)`. Each plane's value is summed in a
/// fixed order, independent of the slab partition. With `COMMIT = false`
/// the currents are left unchanged (read-only evaluation of `i^{n+½}`).
///
/// With [`EnergyMeasure::OpenEms`] the plane value is instead openEMS's
/// measure `ε0·Σ (u^n)² + μ0·Σ (i^{n+½})²` over every edge \[J/m\] (see
/// [`EnergyMeasure`]).
pub(crate) fn energy_h_slab<const COMMIT: bool>(
    store: &FieldStore,
    co: &CoeffRows<'_>,
    s: usize,
    measure: EnergyMeasure,
    sink: &mut dyn FnMut(usize, f64),
) {
    match measure {
        EnergyMeasure::Invariant => h_slab::<true, COMMIT, false>(store, co, s, sink),
        EnergyMeasure::OpenEms => h_slab::<true, COMMIT, true>(store, co, s, sink),
    }
}

fn h_slab<const ENERGY: bool, const COMMIT: bool, const OE: bool>(
    store: &FieldStore,
    co: &CoeffRows<'_>,
    s: usize,
    sink: &mut dyn FnMut(usize, f64),
) {
    let lay = &store.layout;
    let [nx, ny, nz] = lay.n;
    let p = lay.plane;
    let (p0, p1) = lay.planes(s);
    let mut iw = write(&store.i[s]);
    let uo = read(&store.u[s]);
    let un = store.u.get(s + 1).map(read);
    let [ix, iy, iz] = &mut *iw;
    for gi in p0..p1 {
        let l = gi - p0;
        let (ux, uy, uz) = (
            plane(&uo[0], l, p),
            plane(&uo[1], l, p),
            plane(&uo[2], l, p),
        );
        let ix = &mut ix[l * p..(l + 1) * p];
        let iy = &mut iy[l * p..(l + 1) * p];
        let iz = &mut iz[l * p..(l + 1) * p];
        let mut wh = 0.0f64;
        // x: curl_x = (uz[j+1] − uz[j]) − (uy[k+1] − uy[k]).
        let m = nz - 1;
        for j in 0..ny - 1 {
            let r = j * nz;
            wh += hrow::<ENERGY, COMMIT, OE>(
                &mut ix[r..r + m],
                co.row(6, gi, j),
                co.row(9, gi, j),
                &uz[r + nz..],
                &uz[r..],
                &uy[r + 1..],
                &uy[r..],
            );
        }
        // y and z dual edges at the last plane are padding.
        if gi + 1 < nx {
            let next: &Block = if gi + 1 < p1 {
                &uo
            } else {
                un.as_deref().expect("next slab")
            };
            let ln = if gi + 1 < p1 { l + 1 } else { 0 };
            let (uyn, uzn) = (plane(&next[1], ln, p), plane(&next[2], ln, p));
            // y: curl_y = (ux[k+1] − ux[k]) − (uz[i+1] − uz[i]).
            for j in 0..ny {
                let r = j * nz;
                wh += hrow::<ENERGY, COMMIT, OE>(
                    &mut iy[r..r + m],
                    co.row(7, gi, j),
                    co.row(10, gi, j),
                    &ux[r + 1..],
                    &ux[r..],
                    &uzn[r..],
                    &uz[r..],
                );
            }
            // z: curl_z = (uy[i+1] − uy[i]) − (ux[j+1] − ux[j]).
            for j in 0..ny - 1 {
                let r = j * nz;
                wh += hrow::<ENERGY, COMMIT, OE>(
                    &mut iz[r..r + nz],
                    co.row(8, gi, j),
                    co.row(11, gi, j),
                    &uyn[r..],
                    &uy[r..],
                    &ux[r + nz..],
                    &ux[r..],
                );
            }
        }
        if ENERGY {
            let mut we = 0.0f64;
            for (a, uc) in [ux, uy, uz].into_iter().enumerate() {
                let mut acc = [0.0f64; 4];
                for j in 0..ny {
                    let r = j * nz;
                    let (ca, cb) = (co.row(a, gi, j), co.row(3 + a, gi, j));
                    e_energy_row::<OE>(&mut acc, r, &uc[r..r + nz], ca, cb);
                }
                we += (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            sink(gi, if OE { EPS0 * we + MU0 * wh } else { we + wh });
        }
    }
}

/// Ampère update of all E edges in the x-planes of slab `s`:
/// `u ← ca·u + cb·∮i` around the dual face each primary edge pierces.
pub(crate) fn update_e_slab(store: &FieldStore, co: &CoeffRows<'_>, zero: &[f32], s: usize) {
    let lay = &store.layout;
    let [nx, ny, nz] = lay.n;
    let p = lay.plane;
    let (p0, p1) = lay.planes(s);
    let mut uw = write(&store.u[s]);
    let io = read(&store.i[s]);
    let ip = if s > 0 {
        Some(read(&store.i[s - 1]))
    } else {
        None
    };
    let [ux, uy, uz] = &mut *uw;
    let z1 = &zero[..1];
    for gi in p0..p1 {
        let l = gi - p0;
        let (ix, iy, iz) = (
            plane(&io[0], l, p),
            plane(&io[1], l, p),
            plane(&io[2], l, p),
        );
        let (iyp, izp) = if gi == 0 {
            (&zero[..p], &zero[..p])
        } else if gi > p0 {
            (plane(&io[1], l - 1, p), plane(&io[2], l - 1, p))
        } else {
            let prev = ip.as_deref().expect("previous slab");
            let lp = prev[1].len() / p - 1;
            (plane(&prev[1], lp, p), plane(&prev[2], lp, p))
        };
        let ux = &mut ux[l * p..(l + 1) * p];
        let uy = &mut uy[l * p..(l + 1) * p];
        let uz = &mut uz[l * p..(l + 1) * p];
        // x: curl_x = (iz[j] − iz[j−1]) − (iy[k] − iy[k−1]).
        if gi + 1 < nx {
            for j in 0..ny {
                let r = j * nz;
                let (ca, cb) = (co.row(0, gi, j), co.row(3, gi, j));
                let izm = if j == 0 { &zero[..nz] } else { &iz[r - nz..r] };
                e_row(&mut ux[r..r + 1], ca, cb, &iz[r..], izm, &iy[r..], z1);
                e_row(
                    &mut ux[r + 1..r + nz],
                    &ca[1..],
                    &cb[1..],
                    &iz[r + 1..],
                    &izm[1..],
                    &iy[r + 1..],
                    &iy[r..],
                );
            }
        }
        // y: curl_y = (ix[k] − ix[k−1]) − (iz[i] − iz[i−1]).
        for j in 0..ny - 1 {
            let r = j * nz;
            let (ca, cb) = (co.row(1, gi, j), co.row(4, gi, j));
            e_row(&mut uy[r..r + 1], ca, cb, &ix[r..], z1, &iz[r..], &izp[r..]);
            e_row(
                &mut uy[r + 1..r + nz],
                &ca[1..],
                &cb[1..],
                &ix[r + 1..],
                &ix[r..],
                &iz[r + 1..],
                &izp[r + 1..],
            );
        }
        // z: curl_z = (iy[i] − iy[i−1]) − (ix[j] − ix[j−1]).
        let m = nz - 1;
        for j in 0..ny {
            let r = j * nz;
            let ixm = if j == 0 { &zero[..nz] } else { &ix[r - nz..r] };
            e_row(
                &mut uz[r..r + m],
                co.row(2, gi, j),
                co.row(5, gi, j),
                &iy[r..],
                &iyp[r..],
                &ix[r..],
                ixm,
            );
        }
    }
}
