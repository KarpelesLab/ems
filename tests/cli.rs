//! The `ems` binary as a drop-in replacement for the `openEMS` executable:
//! runs the built binary on the fixture models of
//! `tests/fixtures/openems/cli_*` (written by a black-box openEMS, see
//! `scripts/openems-oracle/gen-cli.sh`) in a scratch directory and checks
//! that it writes the same files (names, headers, structure, values within
//! a tolerance), prints the same key console lines and returns the same
//! exit status.
#![cfg(feature = "xml")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use ems::io::openems::{ProbeDomain, read_probe_file};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

/// The `key:` line of a case's `meta.txt`.
fn meta(case: &str, key: &str) -> String {
    let text = std::fs::read_to_string(fixture(case).join("meta.txt")).unwrap();
    let pre = format!("{key}:");
    text.lines()
        .find_map(|l| l.strip_prefix(&pre).map(|v| v.trim().to_string()))
        .unwrap_or_else(|| panic!("{case}/meta.txt: no {key}"))
}

/// A fresh scratch directory with the case's model.
fn scratch(case: &str, variant: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("cli")
        .join(format!("{case}{variant}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixture(case).join("model.xml"), d.join("model.xml")).unwrap();
    d
}

struct Run {
    dir: PathBuf,
    code: i32,
    stdout: String,
    stderr: String,
}

/// Runs `ems` in `dir` with `args`.
fn ems(dir: &Path, args: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_ems"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run the ems binary");
    Run {
        dir: dir.to_path_buf(),
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Runs a fixture case with its recorded openEMS arguments.
fn run_case(case: &str) -> Run {
    let dir = scratch(case, "");
    let args = meta(case, "args");
    let mut a = vec!["model.xml"];
    a.extend(args.split_whitespace());
    let r = ems(&dir, &a);
    eprintln!("{case}: exit {}\n{}{}", r.code, r.stdout, r.stderr);
    r
}

/// The files `ems` wrote (everything but the model).
fn written(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n != "model.xml")
        .collect()
}

fn oracle_files(case: &str) -> BTreeSet<String> {
    meta(case, "files")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = a.iter().map(|x| x * x).sum();
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    }
}

/// Every probe file openEMS wrote for `case` against ems's: kind, header
/// lines after the generator line, rows, sample times, values (relative
/// L2 over all columns).
fn check_probe_files(case: &str, dir: &Path, names: &[&str], tol: f64) {
    for &name in names {
        let a_text = std::fs::read_to_string(fixture(case).join(name)).unwrap();
        let b_text = std::fs::read_to_string(dir.join(name)).unwrap();
        let hdr = |t: &str| -> Vec<String> {
            t.lines()
                .skip(1)
                .filter(|l| l.starts_with('%'))
                .map(str::to_string)
                .collect()
        };
        assert_eq!(hdr(&a_text), hdr(&b_text), "{case}/{name}: header");
        let first = |t: &str| {
            t.lines()
                .next()
                .unwrap()
                .split(" by ")
                .next()
                .unwrap()
                .to_string()
        };
        assert_eq!(first(&a_text), first(&b_text), "{case}/{name}: first line");
        let a = read_probe_file(fixture(case).join(name)).unwrap();
        let b = read_probe_file(dir.join(name)).unwrap();
        assert!(b.generator.starts_with("ems v"), "{name}: {}", b.generator);
        assert_eq!((a.domain, a.kind), (b.domain, b.kind), "{case}/{name}");
        assert_eq!(a.rows.len(), b.rows.len(), "{case}/{name}: rows");
        for (x, y) in a.abscissa().iter().zip(b.abscissa()) {
            let scale = if a.domain == ProbeDomain::Time {
                1e-5
            } else {
                1e-12
            };
            assert!(
                (x - y).abs() <= scale * x.abs().max(1e-12),
                "{case}/{name}: abscissa {x} vs {y}"
            );
        }
        let flat = |f: &ems::io::openems::ProbeFile| -> Vec<f64> {
            f.rows.iter().flat_map(|r| r[1..].to_vec()).collect()
        };
        let e = rel_l2(&flat(&a), &flat(&b));
        eprintln!("{case}/{name}: {} rows, relative L2 {e:.2e}", a.rows.len());
        assert!(e < tol, "{case}/{name}: relative L2 {e:e} > {tol:e}");
    }
}

/// The excitation signal files: same rows, values within 1e-5 of the
/// peak (6 printed digits).
fn check_signals(case: &str, dir: &Path) {
    for f in ["et", "ht"] {
        let rows = |p: PathBuf| -> Vec<(f64, f64)> {
            std::fs::read_to_string(&p)
                .unwrap()
                .lines()
                .map(|l| {
                    let mut it = l.split('\t').map(|v| v.parse::<f64>().unwrap());
                    (it.next().unwrap(), it.next().unwrap())
                })
                .collect()
        };
        let (a, b) = (rows(fixture(case).join(f)), rows(dir.join(f)));
        assert_eq!(a.len(), b.len(), "{case}/{f}: rows");
        let peak = a.iter().fold(0.0f64, |m, r| m.max(r.1.abs()));
        for (x, y) in a.iter().zip(&b) {
            assert!((x.0 - y.0).abs() <= 1e-5 * x.0.max(1e-12), "{case}/{f}: t");
            assert!(
                (x.1 - y.1).abs() <= 2e-5 * peak,
                "{case}/{f}: {x:?} vs {y:?}"
            );
        }
    }
}

/// The value after `key` on the console line that starts with `key`.
fn console_line<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .find_map(|l| l.strip_prefix(key))
        .map(str::trim)
}

/// The VTK XML structure with payloads, ranges, offsets, array pointer
/// names and padding masked.
fn vtk_skeleton(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty()
                && (t.starts_with('<')
                    || !t
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || "+/=_".contains(c)))
        })
        .filter(|l| l.trim().parse::<f64>().is_err())
        .map(|l| {
            let mut s = l.to_string();
            for key in ["RangeMin=\"", "RangeMax=\"", "offset=\"", "Name=\"Array 0x"] {
                if let Some(i) = s.find(key) {
                    let j = s[i + key.len()..].find('"').unwrap() + i + key.len();
                    s.replace_range(i + key.len()..j, "*");
                }
            }
            s.split_whitespace().collect::<Vec<_>>().join(" ")
        })
        .collect()
}

#[cfg(feature = "hdf5-deflate")]
mod payload {
    //! Decoding of VTK's zlib-compressed binary arrays (needs inflate).

    fn b64(s: &str) -> Vec<u8> {
        let val = |c: u8| -> u32 {
            match c {
                b'A'..=b'Z' => u32::from(c - b'A'),
                b'a'..=b'z' => u32::from(c - b'a') + 26,
                b'0'..=b'9' => u32::from(c - b'0') + 52,
                b'+' => 62,
                _ => 63,
            }
        };
        let mut out = Vec::new();
        for ch in s.as_bytes().chunks(4) {
            let n = ch.iter().filter(|&&c| c != b'=').count();
            let mut v = 0u32;
            for (i, &c) in ch.iter().enumerate() {
                if c != b'=' {
                    v |= val(c) << (18 - 6 * i);
                }
            }
            out.extend(&v.to_be_bytes()[1..n.max(1)]);
        }
        out
    }

    /// One encoded array (block header, then the blocks) starting at
    /// `s`; returns the raw bytes and the encoded length.
    pub fn array(s: &str) -> (Vec<u8>, usize) {
        let first = b64(&s[..8]);
        let nb = u32::from_le_bytes(first[..4].try_into().unwrap()) as usize;
        let hlen = (3 + nb) * 4;
        let hb = hlen.div_ceil(3) * 4;
        let head = b64(&s[..hb]);
        let sizes: Vec<usize> = (0..nb)
            .map(|i| u32::from_le_bytes(head[12 + 4 * i..16 + 4 * i].try_into().unwrap()) as usize)
            .collect();
        let total: usize = sizes.iter().sum();
        let db = total.div_ceil(3) * 4;
        let data = b64(&s[hb..hb + db]);
        let mut raw = Vec::new();
        let mut pos = 0;
        for n in sizes {
            raw.extend(
                hdf5rs::filters::deflate::zlib_decompress(&data[pos..pos + n], 1 << 15, 1 << 26)
                    .unwrap(),
            );
            pos += n;
        }
        (raw, hb + db)
    }

    /// The named inline arrays of a VTK XML file, as f64.
    pub fn inline(text: &str) -> Vec<(String, Vec<f64>)> {
        let mut out = Vec::new();
        let mut cur: Option<(String, bool)> = None;
        for l in text.lines() {
            let t = l.trim();
            if t.starts_with("<DataArray") {
                let name = t
                    .split("Name=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap();
                cur = Some((name.to_string(), t.contains("Float64")));
                continue;
            }
            if let Some((name, f64s)) = cur.take()
                && !t.starts_with('<')
            {
                let (raw, _) = array(t);
                let v = if f64s {
                    raw.chunks(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                } else {
                    raw.chunks(4)
                        .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
                        .collect()
                };
                out.push((name, v));
            }
        }
        out
    }

    /// The appended arrays of a VTK XML PolyData file: (name, raw bytes).
    pub fn appended(text: &str) -> Vec<(String, Vec<u8>)> {
        let data = text
            .split("<AppendedData encoding=\"base64\">")
            .nth(1)
            .unwrap();
        let data = data.trim_start().strip_prefix('_').unwrap();
        let data = data.split('<').next().unwrap().trim_end();
        text.lines()
            .filter(|l| l.trim().starts_with("<DataArray"))
            .map(|l| {
                let name = l
                    .split("Name=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap();
                let off: usize = l
                    .split("offset=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                (name.to_string(), array(&data[off..]).0)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------

/// A run with every probe kind and two dumps (`--engine=basic
/// --dump-statistics --debug-boxes --showProbeDiscretization`): the same
/// file names, probe files within 1e-4 (measured 1e-8…4e-6, the H-type
/// files including openEMS's sample after the last step), the same
/// excitation signals, identical box files, the same snapped boxes and key
/// console figures, statistics files of the same layout.
#[test]
fn run_writes_openems_files() {
    let case = "cli_run";
    let r = run_case(case);
    assert_eq!(
        r.code.to_string(),
        meta(case, "exit"),
        "{case}: exit status"
    );
    assert_eq!(written(&r.dir), oracle_files(case), "{case}: files");
    check_probe_files(
        case,
        &r.dir,
        &[
            "ut1", "ut1_FD", "it1", "it1_FD", "ef", "hf", "wv", "wv_FD", "wi",
        ],
        1e-4,
    );
    check_signals(case, &r.dir);
    for f in ["box_dump_.vtk", "box_dump_Et.vtk", "box_dump_Hf.vtk"] {
        assert_eq!(
            std::fs::read_to_string(r.dir.join(f)).unwrap(),
            std::fs::read_to_string(fixture(case).join(f)).unwrap(),
            "{case}/{f}"
        );
    }
    // Console: the snapped boxes and the key figures.
    let oe = std::fs::read_to_string(fixture(case).join("console.txt")).unwrap();
    let snapped = |t: &str| -> Vec<String> {
        t.lines()
            .filter(|l| l.contains(": snapped "))
            .map(str::to_string)
            .collect()
    };
    assert_eq!(
        snapped(&r.stdout),
        snapped(&oe),
        "{case}: probe discretization"
    );
    for key in [
        "FDTD simulation size:",
        "Excitation signal length is:",
        "Max. number of timesteps:",
    ] {
        assert_eq!(
            console_line(&r.stdout, key),
            console_line(&oe, key),
            "{case}: {key}"
        );
    }
    let dt_line = |t: &str| console_line(t, "FDTD timestep is:").unwrap().to_string();
    let (a, b) = (dt_line(&r.stdout), dt_line(&oe));
    // The time step agrees to its printed digits; the Nyquist rate exactly.
    assert_eq!(
        a.split(';').nth(1),
        b.split(';').nth(1),
        "{case}: Nyquist rate"
    );
    // Statistics: same lines, same counts.
    let stats = |p: PathBuf| -> Vec<(f64, String)> {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .map(|l| {
                let (v, d) = l.split_once('\t').unwrap();
                (v.parse().unwrap(), d.to_string())
            })
            .collect()
    };
    let (sa, sb) = (
        stats(fixture(case).join("openEMS_stats.txt")),
        stats(r.dir.join("openEMS_stats.txt")),
    );
    assert_eq!(
        sa.iter().map(|x| &x.1).collect::<Vec<_>>(),
        sb.iter().map(|x| &x.1).collect::<Vec<_>>(),
        "{case}: openEMS_stats.txt lines"
    );
    for k in [0, 2] {
        assert_eq!(sa[k].0, sb[k].0, "{case}: {}", sa[k].1);
    }
    for k in [1, 3] {
        assert!(
            (sa[k].0 / sb[k].0 - 1.0).abs() < 1e-5,
            "{case}: {}",
            sa[k].1
        );
    }
    let head = |p: PathBuf| {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .next()
            .map(str::to_string)
    };
    assert_eq!(
        head(fixture(case).join("openEMS_run_stats.txt")),
        head(r.dir.join("openEMS_run_stats.txt"))
    );
    // Dumps: VTK structure of the kept time steps.
    for f in ["Et_0000000000.vtr", "Et_0000000120.vtr"] {
        let a = std::fs::read_to_string(fixture(case).join(f)).unwrap();
        let b = std::fs::read_to_string(r.dir.join(f)).unwrap();
        assert_eq!(vtk_skeleton(&a), vtk_skeleton(&b), "{case}/{f}");
    }
    #[cfg(feature = "hdf5")]
    {
        // HDF5 dump: the same objects and dataset shapes.
        fn objects(p: &Path) -> Vec<String> {
            fn walk(g: &hdf5rs::Group, path: &str, out: &mut Vec<String>) {
                for name in g.link_names().unwrap() {
                    let p = format!("{}/{name}", path.trim_end_matches('/'));
                    let o = g.object(&name).unwrap();
                    if let Some(sub) = o.as_group() {
                        out.push(p.clone());
                        walk(sub, &p, out);
                    } else if let Some(d) = o.as_dataset() {
                        out.push(format!("{p} {:?}", d.shape()));
                    }
                }
            }
            let f = hdf5rs::File::open(p).unwrap();
            let mut out = Vec::new();
            walk(&f.root().unwrap(), "/", &mut out);
            out.sort();
            out
        }
        assert_eq!(
            objects(&fixture(case).join("Hf.h5")),
            objects(&r.dir.join("Hf.h5")),
            "{case}/Hf.h5"
        );
    }
}

/// `--no-simulation` with the debug options: exit status 1 (as openEMS),
/// the same files; the PEC and material dumps agree (PEC: identical
/// points and lines; materials: `ε`, `μ`, `κ` per edge), the operator dump
/// has openEMS's arrays (values are ems's coefficients), debugCSX.xml
/// holds the same structure.
#[test]
fn debug_files() {
    let case = "cli_debug";
    let r = run_case(case);
    assert_eq!(
        r.code.to_string(),
        meta(case, "exit"),
        "{case}: exit status"
    );
    assert_eq!(written(&r.dir), oracle_files(case), "{case}: files");
    check_signals(case, &r.dir);
    for f in ["PEC_dump.vtp", "material_dump.vtr", "operator_dump.vtr"] {
        let a = std::fs::read_to_string(fixture(case).join(f)).unwrap();
        let b = std::fs::read_to_string(r.dir.join(f)).unwrap();
        assert_eq!(vtk_skeleton(&a), vtk_skeleton(&b), "{case}/{f}: structure");
    }
    // The PEC dump's piece line, padding included.
    let piece = |t: &str| t.lines().find(|l| l.contains("<Piece")).map(str::to_string);
    let read = |d: &Path, f: &str| std::fs::read_to_string(d.join(f)).unwrap();
    assert_eq!(
        piece(&read(&fixture(case), "PEC_dump.vtp")),
        piece(&read(&r.dir, "PEC_dump.vtp"))
    );
    #[cfg(feature = "hdf5-deflate")]
    {
        let a = payload::appended(&read(&fixture(case), "PEC_dump.vtp"));
        let b = payload::appended(&read(&r.dir, "PEC_dump.vtp"));
        assert_eq!(a, b, "{case}: PEC points and lines");
        let a = payload::inline(&read(&fixture(case), "material_dump.vtr"));
        let b = payload::inline(&read(&r.dir, "material_dump.vtr"));
        for ((na, va), (nb, vb)) in a.iter().zip(&b) {
            assert_eq!(na.starts_with("Array"), nb.starts_with("Array"));
            if !na.starts_with("Array") {
                assert_eq!(na, nb);
            }
            let e = va
                .iter()
                .zip(vb)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            eprintln!("{case}/material_dump.vtr {na}: max |diff| {e:.1e}");
            assert!(
                e < 1e-6 * va.iter().fold(1.0f64, |m, x| m.max(x.abs())),
                "{na}"
            );
        }
        let a = payload::inline(&read(&fixture(case), "operator_dump.vtr"));
        let b = payload::inline(&read(&r.dir, "operator_dump.vtr"));
        assert_eq!(a[0].0, "exc_volt");
        assert_eq!(a[0], b[0], "{case}: exc_volt");
    }
    // debugCSX.xml: the model's structure (properties, grid).
    let parse = |d: &Path| {
        csx::ContinuousStructure::from_xml_str(&read(d, "debugCSX.xml")).expect("debugCSX.xml")
    };
    let (a, b) = (parse(&fixture(case)), parse(&r.dir));
    let props = |s: &csx::ContinuousStructure| -> Vec<(String, String, usize)> {
        s.properties()
            .iter()
            .map(|p| {
                (
                    p.name.clone(),
                    p.kind.type_name().to_string(),
                    p.primitives.len(),
                )
            })
            .collect()
    };
    assert_eq!(props(&a), props(&b), "{case}: debugCSX.xml properties");
    for ax in 0..3 {
        assert_eq!(
            a.grid.lines(ax),
            b.grid.lines(ax),
            "{case}: debugCSX.xml grid"
        );
    }
    // Not simulated: no probe files.
    assert!(!r.dir.join("ut1").exists());
}

/// The debug files on cylindrical grids (`cli_debug_cyl`: closed azimuth
/// with the axis, a dielectric, a metal block on the seam, a lumped
/// resistor; `cli_debug_cyl_sector`: a sector on a graded radial mesh):
/// the same files (`.vts` structured grids with Cartesian points, the
/// extra `α_1 + 2π` line of a closed grid), the same PEC edges and points,
/// and the same values wherever both codes describe the same edge: every
/// material average (openEMS's radial weights on the file path,
/// `compile::cylavg`) and every update coefficient (both runs use
/// `TimeStep` 4e-13 s). Masked, as documented in `io::openems::debug`:
/// on a closed grid the `2π` line and the extra line (openEMS: literal
/// angles; ems: the periodic images) and the `α = 0` copies of ρ and z
/// edges (openEMS updates the `2π` copy only); the axis z edge (outside
/// openEMS's dumped operator); H dual edges crossing an outer face
/// (openEMS's `iv` uses the whole dual length) and faces past the last
/// line.
#[test]
fn debug_files_cylindrical() {
    for (case, closed, axis) in [
        ("cli_debug_cyl", true, true),
        ("cli_debug_cyl_sector", false, false),
    ] {
        let r = run_case(case);
        assert_eq!(r.code.to_string(), meta(case, "exit"), "{case}: exit");
        assert_eq!(written(&r.dir), oracle_files(case), "{case}: files");
        let read = |d: &Path, f: &str| std::fs::read_to_string(d.join(f)).unwrap();
        for f in ["PEC_dump.vtp", "material_dump.vts", "operator_dump.vts"] {
            assert_eq!(
                vtk_skeleton(&read(&fixture(case), f)),
                vtk_skeleton(&read(&r.dir, f)),
                "{case}/{f}: structure"
            );
        }
        let piece = |t: &str| t.lines().find(|l| l.contains("<Piece")).map(str::to_string);
        assert_eq!(
            piece(&read(&fixture(case), "PEC_dump.vtp")),
            piece(&read(&r.dir, "PEC_dump.vtp")),
            "{case}: PEC piece"
        );
        #[cfg(feature = "hdf5-deflate")]
        {
            let a = payload::appended(&read(&fixture(case), "PEC_dump.vtp"));
            let b = payload::appended(&read(&r.dir, "PEC_dump.vtp"));
            assert_eq!(a, b, "{case}: PEC points and lines");
            let ext = read(&fixture(case), "material_dump.vts");
            let ext: Vec<usize> = ext
                .split("WholeExtent=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let (nx, ny, nz) = (ext[1] + 1, ext[3] + 1, ext[5] + 1);
            // (array, component, i, j, k) -> compared?
            let keep = |name: &str, c: usize, i: usize, j: usize, k: usize| -> bool {
                let e_side = !matches!(name, "iv" | "ii" | "mue" | "sigma");
                if closed && (j + 2 >= ny || (j == 0 && c != 1 && matches!(name, "vv" | "vi"))) {
                    return false;
                }
                if axis && c == 2 && i == 0 && matches!(name, "vv" | "vi") {
                    return false;
                }
                if !e_side && matches!(name, "iv" | "ii") {
                    let p = [i, j, k];
                    let last = [nx - 1, ny - 1, nz - 1];
                    if p[c] == 0 || p[c] == last[c] {
                        return false;
                    }
                    if (0..3).any(|o| o != c && p[o] == last[o]) {
                        return false;
                    }
                }
                true
            };
            for f in ["material_dump.vts", "operator_dump.vts"] {
                let a = payload::inline(&read(&fixture(case), f));
                let b = payload::inline(&read(&r.dir, f));
                assert_eq!(a.len(), b.len(), "{case}/{f}");
                for ((na, va), (nb, vb)) in a.iter().zip(&b) {
                    assert_eq!(na, nb, "{case}/{f}");
                    assert_eq!(va.len(), vb.len(), "{case}/{f} {na}");
                    let scale = va.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    let (mut worst, mut compared) = (0.0f64, 0usize);
                    for (idx, (x, y)) in va.iter().zip(vb).enumerate() {
                        let (p, c) = (idx / 3, idx % 3);
                        let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
                        if na != "Points" && !keep(na, c, i, j, k) {
                            continue;
                        }
                        compared += 1;
                        worst = worst.max((x - y).abs());
                    }
                    eprintln!(
                        "{case}/{f} {na}: {compared} of {} values compared, max |diff| {worst:.1e} \
                         (scale {scale:.1e})",
                        va.len()
                    );
                    assert!(compared * 3 > va.len(), "{case}/{f} {na}: too few compared");
                    assert!(worst <= 1e-6 * scale.max(1e-30), "{case}/{f} {na}");
                }
            }
        }
    }
}

/// `--disable-dumps`: no dump files, but the boxes of the disabled dumps
/// are still written. (openEMS v0.37 exits with status 1 after such a run,
/// reporting an internal error; ems exits normally.)
#[test]
fn disabled_dumps() {
    let case = "cli_nodumps";
    let r = run_case(case);
    assert_eq!(r.code, 0, "{case}: exit status");
    assert_eq!(meta(case, "exit"), "1", "{case}: the oracle's status");
    assert_eq!(written(&r.dir), oracle_files(case), "{case}: files");
    check_probe_files(case, &r.dir, &["ut1", "wv"], 1e-4);
}

/// The exit status of every invocation recorded from openEMS
/// (`cli_exit/codes.txt`).
#[test]
fn exit_codes() {
    let text = std::fs::read_to_string(fixture("cli_exit").join("codes.txt")).unwrap();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("cli")
        .join("exit");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture("cli_run").join("model.xml"), dir.join("model.xml")).unwrap();
    let mut n = 0;
    for l in text.lines().filter(|l| !l.starts_with('#')) {
        let mut f = l.split('\t');
        let (label, code) = (f.next().unwrap(), f.next().unwrap());
        let args: Vec<&str> = f.next().unwrap_or("").split_whitespace().collect();
        let r = ems(&dir, &args);
        assert_eq!(
            r.code.to_string(),
            code,
            "{label} ({args:?}):\n{}{}",
            r.stdout,
            r.stderr
        );
        n += 1;
    }
    assert!(n >= 8);
}

/// `ems --help` lists every option `openEMS --help` lists.
#[test]
fn help_lists_openems_options() {
    let r = ems(Path::new(env!("CARGO_TARGET_TMPDIR")), &["--help"]);
    assert_eq!(r.code, 0);
    let opts = std::fs::read_to_string(fixture("cli_exit").join("options.txt")).unwrap();
    for o in opts.lines() {
        assert!(
            r.stdout.contains(o),
            "--help does not list {o}:\n{}",
            r.stdout
        );
    }
    for o in ["-v", "-vv", "-vvv"] {
        assert!(r.stdout.contains(o));
    }
}

/// The accepted option syntaxes give the same run: `--opt=value`,
/// `--opt value`, abbreviations, and the verbose forms.
#[test]
fn option_syntaxes() {
    let dir = scratch("cli_run", "_syntax");
    let r = ems(
        &dir,
        &[
            "model.xml",
            "--engine",
            "multithreaded",
            "--numThreads",
            "2",
            "--no-sim",
            "-vvv",
            "--exact-endcriteria",
        ],
    );
    assert_eq!(
        r.code, 1,
        "--no-simulation exits with 1:\n{}{}",
        r.stdout, r.stderr
    );
    assert!(r.stdout.contains("verbose level 3"), "{}", r.stdout);
    assert!(r.stdout.contains("fixed number of threads: 2"));
    assert!(r.stdout.contains("Engine threads\t\t: 2"), "{}", r.stdout);
    // Preprocessing writes the signals only.
    assert_eq!(
        written(&dir),
        ["et", "ht"].into_iter().map(String::from).collect()
    );
    let r = ems(
        &dir,
        &[
            "model.xml",
            "--verbose=2",
            "--engine=gpu",
            "--no-simulation",
        ],
    );
    assert_eq!(r.code, 1);
    assert!(r.stdout.contains("verbose level 2"));
    assert!(r.stdout.contains("engine 'gpu' requested"));
}
