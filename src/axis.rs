//! Cartesian coordinate axes.

use core::fmt;

/// A Cartesian axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// The x axis (index 0).
    X,
    /// The y axis (index 1).
    Y,
    /// The z axis (index 2).
    Z,
}

impl Axis {
    /// All axes in index order.
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    /// Index of the axis: x = 0, y = 1, z = 2.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The axis with index `i` (0, 1 or 2).
    pub const fn from_index(i: usize) -> Option<Axis> {
        match i {
            0 => Some(Axis::X),
            1 => Some(Axis::Y),
            2 => Some(Axis::Z),
            _ => None,
        }
    }

    /// The next axis in cyclic order (x → y → z → x).
    pub const fn next(self) -> Axis {
        match self {
            Axis::X => Axis::Y,
            Axis::Y => Axis::Z,
            Axis::Z => Axis::X,
        }
    }

    /// The previous axis in cyclic order (x → z → y → x).
    pub const fn prev(self) -> Axis {
        match self {
            Axis::X => Axis::Z,
            Axis::Y => Axis::X,
            Axis::Z => Axis::Y,
        }
    }

    /// The two other axes `(next, prev)` in cyclic order, so that
    /// `self × next = prev` forms a right-handed triple, e.g. `X → (Y, Z)`.
    pub const fn others(self) -> (Axis, Axis) {
        (self.next(), self.prev())
    }

    /// Lower-case name: `"x"`, `"y"` or `"z"`.
    pub const fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_round_trip() {
        for (i, a) in Axis::ALL.into_iter().enumerate() {
            assert_eq!(a.index(), i);
            assert_eq!(Axis::from_index(i), Some(a));
        }
        assert_eq!(Axis::from_index(3), None);
    }

    #[test]
    fn cyclic_order() {
        assert_eq!(Axis::X.others(), (Axis::Y, Axis::Z));
        assert_eq!(Axis::Y.others(), (Axis::Z, Axis::X));
        assert_eq!(Axis::Z.others(), (Axis::X, Axis::Y));
        for a in Axis::ALL {
            assert_eq!(a.next().prev(), a);
            assert_eq!(a.next().next().next(), a);
        }
        assert_eq!(Axis::Y.to_string(), "y");
    }
}
