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

use super::coeffs::Coefficients;
use super::fields::{Block, FieldStore, read, write};

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
fn flush(x: f32) -> f32 {
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
pub(crate) fn update_h_slab(store: &FieldStore, co: &Coefficients, s: usize) {
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
        let g = gi * p..(gi + 1) * p;
        let (ux, uy, uz) = (
            plane(&uo[0], l, p),
            plane(&uo[1], l, p),
            plane(&uo[2], l, p),
        );
        let ix = &mut ix[l * p..(l + 1) * p];
        let iy = &mut iy[l * p..(l + 1) * p];
        let iz = &mut iz[l * p..(l + 1) * p];
        // x: curl_x = (uz[j+1] − uz[j]) − (uy[k+1] − uy[k]).
        let (da, db) = (&co.data[6][g.clone()], &co.data[9][g.clone()]);
        let m = nz - 1;
        for j in 0..ny - 1 {
            let r = j * nz;
            h_row(
                &mut ix[r..r + m],
                &da[r..],
                &db[r..],
                &uz[r + nz..],
                &uz[r..],
                &uy[r + 1..],
                &uy[r..],
            );
        }
        if gi + 1 == nx {
            continue; // y and z dual edges at the last plane are padding
        }
        let next: &Block = if gi + 1 < p1 {
            &uo
        } else {
            un.as_deref().expect("next slab")
        };
        let ln = if gi + 1 < p1 { l + 1 } else { 0 };
        let (uyn, uzn) = (plane(&next[1], ln, p), plane(&next[2], ln, p));
        // y: curl_y = (ux[k+1] − ux[k]) − (uz[i+1] − uz[i]).
        let (da, db) = (&co.data[7][g.clone()], &co.data[10][g.clone()]);
        for j in 0..ny {
            let r = j * nz;
            h_row(
                &mut iy[r..r + m],
                &da[r..],
                &db[r..],
                &ux[r + 1..],
                &ux[r..],
                &uzn[r..],
                &uz[r..],
            );
        }
        // z: curl_z = (uy[i+1] − uy[i]) − (ux[j+1] − ux[j]).
        let (da, db) = (&co.data[8][g.clone()], &co.data[11][g]);
        for j in 0..ny - 1 {
            let r = j * nz;
            h_row(
                &mut iz[r..r + nz],
                &da[r..],
                &db[r..],
                &uyn[r..],
                &uy[r..],
                &ux[r + nz..],
                &ux[r..],
            );
        }
    }
}

/// Ampère update of all E edges in the x-planes of slab `s`:
/// `u ← ca·u + cb·∮i` around the dual face each primary edge pierces.
pub(crate) fn update_e_slab(store: &FieldStore, co: &Coefficients, zero: &[f32], s: usize) {
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
        let g = gi * p..(gi + 1) * p;
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
            let (ca, cb) = (&co.data[0][g.clone()], &co.data[3][g.clone()]);
            for j in 0..ny {
                let r = j * nz;
                let izm = if j == 0 { &zero[..nz] } else { &iz[r - nz..r] };
                e_row(
                    &mut ux[r..r + 1],
                    &ca[r..],
                    &cb[r..],
                    &iz[r..],
                    izm,
                    &iy[r..],
                    z1,
                );
                e_row(
                    &mut ux[r + 1..r + nz],
                    &ca[r + 1..],
                    &cb[r + 1..],
                    &iz[r + 1..],
                    &izm[1..],
                    &iy[r + 1..],
                    &iy[r..],
                );
            }
        }
        // y: curl_y = (ix[k] − ix[k−1]) − (iz[i] − iz[i−1]).
        let (ca, cb) = (&co.data[1][g.clone()], &co.data[4][g.clone()]);
        for j in 0..ny - 1 {
            let r = j * nz;
            e_row(
                &mut uy[r..r + 1],
                &ca[r..],
                &cb[r..],
                &ix[r..],
                z1,
                &iz[r..],
                &izp[r..],
            );
            e_row(
                &mut uy[r + 1..r + nz],
                &ca[r + 1..],
                &cb[r + 1..],
                &ix[r + 1..],
                &ix[r..],
                &iz[r + 1..],
                &izp[r + 1..],
            );
        }
        // z: curl_z = (iy[i] − iy[i−1]) − (ix[j] − ix[j−1]).
        let (ca, cb) = (&co.data[2][g.clone()], &co.data[5][g]);
        let m = nz - 1;
        for j in 0..ny {
            let r = j * nz;
            let ixm = if j == 0 { &zero[..nz] } else { &ix[r - nz..r] };
            e_row(
                &mut uz[r..r + m],
                &ca[r..],
                &cb[r..],
                &iy[r..],
                &iyp[r..],
                &ix[r..],
                ixm,
            );
        }
    }
}
