//! Minimal streaming base64 encoder (RFC 4648 §4, standard alphabet, with
//! `=` padding).
//!
//! Used for the inline `binary` encoding of VTK XML data arrays. Only
//! encoding is needed by the crate; a decoder lives in the tests.

use std::io::{self, Write};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Size of the internal output buffer, in bytes.
const OUT_CAP: usize = 8192;

/// Encodes one group of one to three input bytes into four output characters.
fn encode_group(input: &[u8]) -> [u8; 4] {
    let b0 = input[0];
    let b1 = input.get(1).copied().unwrap_or(0);
    let b2 = input.get(2).copied().unwrap_or(0);
    let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
    let c = |shift: u32| ALPHABET[((n >> shift) & 63) as usize];
    [
        c(18),
        c(12),
        if input.len() > 1 { c(6) } else { b'=' },
        if input.len() > 2 { c(0) } else { b'=' },
    ]
}

/// Encodes `input` as a base64 string.
pub(crate) fn encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity(input.len().div_ceil(3) * 4);
    let mut enc = Encoder::new(&mut out);
    enc.write_bytes(input)
        .expect("writing to a Vec cannot fail");
    enc.finish().expect("writing to a Vec cannot fail");
    String::from_utf8(out).expect("base64 output is ASCII")
}

/// Streaming base64 encoder.
///
/// Bytes pushed with [`Encoder::write_bytes`] are encoded as one continuous
/// stream, so the output is identical to encoding the concatenation of all
/// inputs at once. [`Encoder::finish`] must be called to emit the final
/// (padded) group and flush the internal buffer.
pub(crate) struct Encoder<W: Write> {
    inner: W,
    pending: [u8; 3],
    npending: usize,
    out: Vec<u8>,
}

impl<W: Write> Encoder<W> {
    /// Creates an encoder writing characters to `inner`.
    pub(crate) fn new(inner: W) -> Self {
        Encoder {
            inner,
            pending: [0; 3],
            npending: 0,
            out: Vec::with_capacity(OUT_CAP + 4),
        }
    }

    fn push_group(&mut self, group: &[u8]) -> io::Result<()> {
        self.out.extend_from_slice(&encode_group(group));
        if self.out.len() >= OUT_CAP {
            self.inner.write_all(&self.out)?;
            self.out.clear();
        }
        Ok(())
    }

    /// Appends `bytes` to the encoded stream.
    pub(crate) fn write_bytes(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        if self.npending > 0 {
            let take = (3 - self.npending).min(bytes.len());
            self.pending[self.npending..self.npending + take].copy_from_slice(&bytes[..take]);
            self.npending += take;
            bytes = &bytes[take..];
            if self.npending < 3 {
                return Ok(());
            }
            let group = self.pending;
            self.push_group(&group)?;
            self.npending = 0;
        }
        let (groups, rest) = bytes.as_chunks::<3>();
        for group in groups {
            self.push_group(group)?;
        }
        self.pending[..rest.len()].copy_from_slice(rest);
        self.npending = rest.len();
        Ok(())
    }

    /// Emits the final group (with padding) and flushes the internal buffer.
    /// Returns the inner writer.
    pub(crate) fn finish(mut self) -> io::Result<W> {
        if self.npending > 0 {
            let group = self.pending;
            let n = self.npending;
            self.npending = 0;
            self.push_group(&group[..n])?;
        }
        self.inner.write_all(&self.out)?;
        self.out.clear();
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc4648_vectors() {
        // RFC 4648 §10 test vectors.
        let cases = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (input, expected) in cases {
            assert_eq!(encode(input.as_bytes()), expected, "input {input:?}");
        }
    }

    #[test]
    fn full_alphabet_edges() {
        assert_eq!(encode(&[0xff, 0xff, 0xff]), "////");
        assert_eq!(encode(&[0xfb, 0xef, 0xbe]), "++++");
        assert_eq!(encode(&[0, 0, 0]), "AAAA");
        assert_eq!(encode(&[0xff]), "/w==");
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let expected = encode(&data);
        assert_eq!(expected.len(), data.len().div_ceil(3) * 4);
        // Split the input at awkward positions.
        for split in [1usize, 2, 4, 5, 7, 1000] {
            let mut out = Vec::new();
            let mut enc = Encoder::new(&mut out);
            for chunk in data.chunks(split) {
                enc.write_bytes(chunk).unwrap();
            }
            enc.finish().unwrap();
            assert_eq!(String::from_utf8(out).unwrap(), expected, "split {split}");
        }
    }
}
