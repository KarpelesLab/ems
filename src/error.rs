//! Error type.

use core::fmt;

/// Errors produced by this crate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Error reported by the model layer ([`csx`]).
    Csx(csx::Error),
    /// The model or solver settings are inconsistent or out of range.
    Invalid(String),
    /// The model uses a feature this solver does not implement (yet).
    Unsupported(String),
    /// I/O failure (message of the underlying error).
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Csx(e) => write!(f, "model error: {e}"),
            Error::Invalid(m) => write!(f, "invalid setup: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Io(m) => write!(f, "I/O error: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Csx(e) => Some(e),
            _ => None,
        }
    }
}

impl From<csx::Error> for Error {
    fn from(e: csx::Error) -> Self {
        Error::Csx(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Convenience alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;
