//! Field storage, split into x-slabs for the worker pool.
//!
//! Each of the six field arrays is logically one `nx·ny·nz` array in the
//! layout described in the [engine module docs](super). Physically it is cut
//! into contiguous x-slabs (whole `ny·nz` planes), one per worker thread, and
//! each slab's three E (or three H) components sit behind one `RwLock`. This
//! lets every worker hold a write lock on the slab it updates and read locks
//! on the slabs it only reads, without `unsafe`. The partition never
//! changes the arithmetic, so results do not depend on it.

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::axis::Axis;

/// The three components of one field type (E or H) of one slab.
pub(crate) type Block = [Vec<f32>; 3];

pub(crate) fn read<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn write<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(PoisonError::into_inner)
}

/// Array shape and slab partition.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub(crate) n: [usize; 3],
    /// `ny·nz`.
    pub(crate) plane: usize,
    /// First x-plane of each slab, plus `nx` at the end.
    pub(crate) slab_start: Vec<usize>,
    /// Slab of each x-plane.
    plane_slab: Vec<usize>,
}

impl Layout {
    pub(crate) fn new(n: [usize; 3], slabs: usize) -> Self {
        let slabs = slabs.clamp(1, n[0]);
        let (q, r) = (n[0] / slabs, n[0] % slabs);
        let mut slab_start = Vec::with_capacity(slabs + 1);
        let mut p = 0;
        for s in 0..slabs {
            slab_start.push(p);
            p += q + usize::from(s < r);
        }
        slab_start.push(n[0]);
        let mut plane_slab = vec![0; n[0]];
        for s in 0..slabs {
            plane_slab[slab_start[s]..slab_start[s + 1]].fill(s);
        }
        Self {
            n,
            plane: n[1] * n[2],
            slab_start,
            plane_slab,
        }
    }

    pub(crate) fn slabs(&self) -> usize {
        self.slab_start.len() - 1
    }

    pub(crate) fn planes(&self, s: usize) -> (usize, usize) {
        (self.slab_start[s], self.slab_start[s + 1])
    }

    /// Slab and offset inside the slab block of node `[i, j, k]`.
    #[inline]
    pub(crate) fn locate(&self, [i, j, k]: [usize; 3]) -> (usize, usize) {
        assert!(
            i < self.n[0] && j < self.n[1] && k < self.n[2],
            "node [{i}, {j}, {k}] out of range {:?}",
            self.n
        );
        let s = self.plane_slab[i];
        (
            s,
            ((i - self.slab_start[s]) * self.n[1] + j) * self.n[2] + k,
        )
    }
}

/// The six field arrays, slab-partitioned.
#[derive(Debug)]
pub(crate) struct FieldStore {
    pub(crate) layout: Layout,
    pub(crate) u: Vec<RwLock<Block>>,
    pub(crate) i: Vec<RwLock<Block>>,
}

impl FieldStore {
    pub(crate) fn new(layout: Layout) -> Self {
        let mk = |s: usize| {
            let (a, b) = layout.planes(s);
            let len = (b - a) * layout.plane;
            RwLock::new([vec![0.0; len], vec![0.0; len], vec![0.0; len]])
        };
        let u = (0..layout.slabs()).map(mk).collect();
        let i = (0..layout.slabs()).map(mk).collect();
        Self { layout, u, i }
    }

    /// Locks everything for exclusive access (operators, user access).
    pub(crate) fn lock(&self) -> Fields<'_> {
        Fields {
            u: FieldArrays {
                layout: &self.layout,
                blocks: self.u.iter().map(write).collect(),
            },
            i: FieldArrays {
                layout: &self.layout,
                blocks: self.i.iter().map(write).collect(),
            },
        }
    }
}

/// Exclusive view of one field type: the E edge voltages `u` or the H
/// dual-edge currents `i`, all three components.
///
/// Values are addressed by component [`Axis`] and node `[i, j, k]` in
/// the convention of the [engine module docs](super). Whole x-planes
/// (`ny·nz` values, z contiguous) are available as slices for bulk
/// operators.
#[derive(Debug)]
pub struct FieldArrays<'a> {
    layout: &'a Layout,
    blocks: Vec<RwLockWriteGuard<'a, Block>>,
}

impl FieldArrays<'_> {
    /// Node counts `[nx, ny, nz]`.
    pub fn nodes(&self) -> [usize; 3] {
        self.layout.n
    }

    /// Value of component `axis` at `node`.
    ///
    /// # Panics
    /// If the node is out of range.
    #[inline]
    pub fn get(&self, axis: Axis, node: [usize; 3]) -> f32 {
        let (s, o) = self.layout.locate(node);
        self.blocks[s][axis.index()][o]
    }

    /// Mutable reference to component `axis` at `node`.
    ///
    /// Writing padding entries (edges that do not exist, see the engine
    /// docs) is allowed but meaningless; keep them at zero.
    ///
    /// # Panics
    /// If the node is out of range.
    #[inline]
    pub fn get_mut(&mut self, axis: Axis, node: [usize; 3]) -> &mut f32 {
        let (s, o) = self.layout.locate(node);
        &mut self.blocks[s][axis.index()][o]
    }

    /// Sets component `axis` at `node`.
    #[inline]
    pub fn set(&mut self, axis: Axis, node: [usize; 3], v: f32) {
        *self.get_mut(axis, node) = v;
    }

    /// The storage layout (for precomputed locations, see
    /// [`at`](Self::at)).
    pub(crate) fn layout(&self) -> &Layout {
        self.layout
    }

    /// Component `axis` at a location `(slab, offset)` from
    /// [`Layout::locate`].
    #[inline]
    pub(crate) fn at(&self, axis: Axis, (s, o): (usize, usize)) -> f32 {
        self.blocks[s][axis.index()][o]
    }

    /// Sets component `axis` at a location from [`Layout::locate`].
    #[inline]
    pub(crate) fn set_at(&mut self, axis: Axis, (s, o): (usize, usize), v: f32) {
        self.blocks[s][axis.index()][o] = v;
    }

    /// The x-plane `i` of component `axis` (`ny·nz` values, index
    /// `j·nz + k`).
    pub fn plane(&self, axis: Axis, i: usize) -> &[f32] {
        let (s, o) = self.layout.locate([i, 0, 0]);
        &self.blocks[s][axis.index()][o..o + self.layout.plane]
    }

    /// Mutable x-plane `i` of component `axis`.
    pub fn plane_mut(&mut self, axis: Axis, i: usize) -> &mut [f32] {
        let (s, o) = self.layout.locate([i, 0, 0]);
        &mut self.blocks[s][axis.index()][o..o + self.layout.plane]
    }

    /// Copies component `axis` into one contiguous array in global layout.
    pub fn to_vec(&self, axis: Axis) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.layout.n[0] * self.layout.plane);
        for b in &self.blocks {
            v.extend_from_slice(&b[axis.index()]);
        }
        v
    }

    /// Sets every value of every component to zero.
    pub fn clear(&mut self) {
        for b in &mut self.blocks {
            for c in b.iter_mut() {
                c.fill(0.0);
            }
        }
    }
}

/// Exclusive access to all fields, handed to [operators](super::Operator)
/// and returned by [`Engine::fields_mut`](super::Engine::fields_mut).
///
/// `u` and `i` are separate public fields so an operator can read one while
/// writing the other.
#[derive(Debug)]
pub struct Fields<'a> {
    /// E edge voltages `u = ∫E·dl` \[V\] (three components).
    pub u: FieldArrays<'a>,
    /// H dual-edge currents `i = ∫H·dl` \[A\] (three components).
    pub i: FieldArrays<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition() {
        let l = Layout::new([10, 3, 4], 4);
        assert_eq!(l.slab_start, vec![0, 3, 6, 8, 10]);
        // Plane 7 is local plane 1 of slab 2: ((1·3) + 1)·4 + 2.
        assert_eq!(l.locate([7, 1, 2]), (2, 18));
        let l = Layout::new([3, 2, 2], 8);
        assert_eq!(l.slabs(), 3);
        let store = FieldStore::new(Layout::new([5, 2, 3], 2));
        let mut f = store.lock();
        f.u.set(Axis::Y, [4, 1, 2], 3.0);
        f.i.set(Axis::X, [0, 0, 1], 2.0);
        assert_eq!(f.u.get(Axis::Y, [4, 1, 2]), 3.0);
        let v = f.u.to_vec(Axis::Y);
        assert_eq!(v[(4 * 2 + 1) * 3 + 2], 3.0);
        assert_eq!(f.i.plane(Axis::X, 0)[1], 2.0);
    }
}
