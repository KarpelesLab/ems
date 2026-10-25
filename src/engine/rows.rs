//! Row-deduplicated view of the update coefficients for the kernels.
//!
//! On large grids the kernels are limited by memory bandwidth, and the
//! twelve coefficient arrays are 48 of the ~120 bytes moved per cell and
//! step. Many z-rows of a coefficient array are bitwise identical: `ca` and
//! `da` are exactly `1.0` everywhere in lossless media, and `cb`, `db` only
//! depend on the material and the local cell sizes, which repeat across a
//! uniformly meshed region. [`KernelCoeffs`] stores each array either as is
//! or, when few of its rows are distinct, as a small pool of distinct rows
//! plus one index per row. The pool stays in cache, so the kernels then
//! stream only the fields.
//!
//! The values the kernels read are exactly those of [`Coefficients`], so
//! results are unchanged bit for bit.

use std::collections::HashMap;

use super::coeffs::Coefficients;

/// An array is pooled when at most `1/POOL_RATIO` of its rows are distinct
/// (bounds the extra memory to a quarter of the array plus the index).
const POOL_RATIO: usize = 4;

/// Storage of one coefficient array.
#[derive(Debug, Clone)]
enum Rows {
    /// Read the rows from [`Coefficients`] directly.
    Direct,
    /// Distinct rows, `nz` values each, and the pool row of every
    /// `(i, j)` row (index `i·ny + j`).
    Pooled { pool: Vec<f32>, index: Vec<u32> },
}

/// The kernel representation of a [`Coefficients`] set (see the module
/// docs). Rebuilt whenever the coefficients may have changed.
#[derive(Debug, Clone)]
pub(crate) struct KernelCoeffs {
    n: [usize; 3],
    rows: [Rows; 12],
}

impl KernelCoeffs {
    /// Analyses `co` (the twelve arrays in parallel when `threads > 1`).
    pub(crate) fn new(co: &Coefficients, threads: usize) -> Self {
        let n = co.nodes();
        let build = |a: usize| pool_rows(&co.data[a], n[2]);
        let rows: [Rows; 12] = if threads > 1 {
            std::thread::scope(|s| {
                let hs: Vec<_> = (0..12).map(|a| s.spawn(move || build(a))).collect();
                let mut it = hs.into_iter().map(|h| h.join().expect("row analysis"));
                core::array::from_fn(|_| it.next().expect("12 arrays"))
            })
        } else {
            core::array::from_fn(build)
        };
        Self { n, rows }
    }

    /// Binds the view to the coefficients it was built from; `oe_curl`
    /// selects openEMS's summation order of the circulations
    /// ([`CurlOrder::OpenEms`](super::CurlOrder::OpenEms)).
    pub(crate) fn view<'a>(&'a self, co: &'a Coefficients, oe_curl: bool) -> CoeffRows<'a> {
        debug_assert_eq!(co.nodes(), self.n);
        CoeffRows {
            co,
            kc: self,
            oe_curl,
        }
    }

    /// Number of arrays stored as pools (diagnostics, tests).
    #[cfg(test)]
    fn pooled(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| matches!(r, Rows::Pooled { .. }))
            .count()
    }
}

/// Pools the `nz`-value rows of `v` if few are distinct.
fn pool_rows(v: &[f32], nz: usize) -> Rows {
    let nrows = v.len() / nz;
    let limit = nrows / POOL_RATIO;
    let mut pool: Vec<f32> = Vec::new();
    let mut index = Vec::with_capacity(nrows);
    // Hash of the row bits → pool rows with that hash.
    let mut seen: HashMap<u64, Vec<u32>> = HashMap::new();
    for row in v.chunks_exact(nz) {
        let h = hash(row);
        let cands = seen.entry(h).or_default();
        let hit = cands.iter().copied().find(|&c| {
            let p = &pool[c as usize * nz..(c as usize + 1) * nz];
            p.iter().zip(row).all(|(a, b)| a.to_bits() == b.to_bits())
        });
        let id = match hit {
            Some(c) => c,
            None => {
                let c = (pool.len() / nz) as u32;
                if c as usize >= limit.max(1) {
                    return Rows::Direct;
                }
                pool.extend_from_slice(row);
                cands.push(c);
                c
            }
        };
        index.push(id);
    }
    pool.shrink_to_fit();
    Rows::Pooled { pool, index }
}

/// FNV-1a over the bit patterns.
fn hash(row: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in row {
        h = (h ^ u64::from(x.to_bits())).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Row access to the coefficients for the kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoeffRows<'a> {
    co: &'a Coefficients,
    kc: &'a KernelCoeffs,
    oe_curl: bool,
}

impl<'a> CoeffRows<'a> {
    /// Whether the kernels sum the circulations in openEMS's order
    /// ([`CurlOrder::OpenEms`](super::CurlOrder::OpenEms)).
    #[inline(always)]
    pub(crate) fn oe_curl(&self) -> bool {
        self.oe_curl
    }

    /// Row `(gi, j)` (`nz` values, k = 0..nz) of array `a` (ca 0–2,
    /// cb 3–5, da 6–8, db 9–11).
    #[inline(always)]
    pub(crate) fn row(&self, a: usize, gi: usize, j: usize) -> &'a [f32] {
        let [_, ny, nz] = self.kc.n;
        let r = gi * ny + j;
        match &self.kc.rows[a] {
            Rows::Direct => &self.co.data[a][r * nz..(r + 1) * nz],
            Rows::Pooled { pool, index } => {
                let o = index[r] as usize * nz;
                &pool[o..o + nz]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EdgeMaterials, FaceConditions, Grid};

    #[test]
    fn rows_match_and_pool() {
        let g = Grid::new([
            vec![0.0, 1.0, 2.0, 3.5, 5.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.5, 1.0, 2.0, 3.0],
        ])
        .unwrap();
        let co = Coefficients::from_edges(
            &g,
            &EdgeMaterials::vacuum(&g),
            &FaceConditions::default(),
            1e-12,
        )
        .unwrap();
        for threads in [1, 4] {
            let kc = KernelCoeffs::new(&co, threads);
            // ca, da are 0/1 patterns: pooled.
            assert!(kc.pooled() >= 6, "{}", kc.pooled());
            let v = kc.view(&co, false);
            let [nx, ny, nz] = co.nodes();
            for a in 0..12 {
                for gi in 0..nx {
                    for j in 0..ny {
                        let o = (gi * ny + j) * nz;
                        let want = &co.data[a][o..o + nz];
                        let got = v.row(a, gi, j);
                        assert!(
                            want.iter()
                                .zip(got)
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn distinct_rows_stay_direct() {
        let v: Vec<f32> = (0..64).map(|x| x as f32).collect();
        assert!(matches!(pool_rows(&v, 4), Rows::Direct));
        let v = vec![1.0f32; 64];
        match pool_rows(&v, 4) {
            Rows::Pooled { pool, index } => {
                assert_eq!(pool.len(), 4);
                assert!(index.iter().all(|&i| i == 0));
            }
            Rows::Direct => panic!("uniform rows not pooled"),
        }
    }
}
