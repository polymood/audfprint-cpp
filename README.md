# audfprint-cpp

A reimplementation of [audfprint](https://github.com/dpwe/audfprint) (Dan Ellis's
landmark-based audio fingerprinter, originally Python) in C++. Same CLI, same algorithm,
same output, same database files, much faster. All credit for the method goes to the
original project; this is an independent port, not affiliated with it.

## Feature parity with the Python version

| feature | status |
|---|---|
| `new`, `add`, `match`, `precompute`, `merge`, `newmerge`, `list`, `remove` | yes, output identical |
| all analysis/matching options (`-n -h -b -t -u -r -p -i -w -N -x -X -R -Q -S -F -P -D -K -k -C -l -T -W -V -o`) | yes, output identical |
| `.pklz` databases (gzipped pickle) | read and write; interchangeable with Python audfprint (numpy 1.x and 2.x), including old-version pickles |
| `.afpt` / `.afpk` precomputed fingerprint files | read and write, byte-identical |
| `-H/--ncores` parallelism | yes (threads; default all cores instead of 1) |
| Matlab `.mat` databases | not supported |
| `-I/--illustrate`, `-J/--illustrate-hpf` (matplotlib plots) | not ported |
| `dpwe_builder.py` / `dpwe_matcher.py` (MIREX wrappers), Python module API | not ported |
| random eviction when a hash bucket overflows | same rule, different RNG, so very full DBs can drift slightly |

## Build

```
make
```

Needs g++ (C++20), glibc libmvec, zlib, and FFmpeg 8.1.3 runtime libs (`libavformat.so.62`,
`libavcodec.so.62`, `libswresample.so.6`, `libavutil.so.60`). The FFmpeg public headers for
exactly that version are vendored in `third_party/ffsrc` (no `-devel` package needed; LGPL
2.1, see `third_party/ffsrc/COPYING.LGPLv2.1`). pocketfft (the FFT numpy uses, BSD-3) is
vendored in `third_party/`.

Code is MIT, like the original audfprint (see `LICENSE`).

## Usage

Same as the Python CLI, see `./audfprint --help`:

```
./audfprint new   --dbase fpdbase.pklz tracks/*.mp3
./audfprint add   --dbase fpdbase.pklz more/*.mp3
./audfprint match --dbase fpdbase.pklz query.mp3
```

Database format is chosen by name: a `.pklz` name writes the Python-compatible gzipped
pickle; any other name writes a compact native format that loads much faster (no 400 MB
table to inflate). Reading detects the format from the file contents.

The "Processed ... x RT" line reports wall time (Python reports CPU time of the main process).

## Parity test

```
tests/parity.sh <path/to/python/audfprint> <some-long-audio-file> [python]
```

Builds MP3/WAV/FLAC tracks and queries from the audio file, runs both implementations and
compares: precompute files byte for byte, `match`/`list` output across option sets, after
`new`/`add`/`merge`/`newmerge`/`remove`, precomputed inputs, `.pklz` files written by either
side, and protocol-2 / old-version pickles. Checked against numpy 2.5.4 / scipy 1.18.1 /
ffmpeg 8.1.3 (and numpy 1.26 for reading C++-written `.pklz`): all identical. Decoded PCM was
also checked byte-identical to `ffmpeg -i f -f s16le -ac 1 -ar 11025 -` at 1 to 24 threads.

## How it's fast

- In-process FFmpeg decode. MP3 is split into packet chunks decoded and resampled in
  parallel, overlapped with demuxing. Each chunk starts a few packets early to rebuild decoder
  state, and resamples from a filter-phase-aligned sample with history and lookahead, which
  is why the result is bit-exact. Frame sizes get verified; anything unexpected falls back to
  one sequential pass (also the path for non-MP3 codecs).
- Spectrogram: batched SIMD pocketfft, AVX2 vector log, threaded HPF; peak picking with
  branch-free AVX2 candidate masks; huge-page buffers; a persistent nested thread pool.
- Match-mode time shifts are analysed in parallel; files in a batch are spread over cores.
- `.pklz` writing compresses the table as parallel gzip members (Python's gzip reads them).

## Benchmark (CIA.mp3, 265.7 s, 320 kbps MP3; Ryzen 9 5900X, 12C/24T)

Wall time per CLI invocation (hyperfine, mean of 5; batch rows mean of 2).

| command | Python | C++ | speedup |
|---|---|---|---|
| `precompute CIA.mp3` | 1.180 s | 0.114 s | 10x |
| `new CIA.mp3`, `.pklz` | 2.608 s | 0.212 s | 12x |
| `new CIA.mp3`, native db | | 0.128 s | 20x |
| `match` 15 s query, `.pklz` | 2.316 s | 0.313 s | 7.4x |
| `match` 15 s query, native db | | 0.076 s | 30x |
| `match CIA.mp3` (full, 4 shifts), `.pklz` | 3.625 s | 0.412 s | 8.8x |
| `match CIA.mp3` (full, 4 shifts), native db | | 0.173 s | 21x |
| `new` 32 files, `.pklz` (Python `-H 24`) | 29.2 s | 1.31 s | 22x |
| `match` 32 full-length queries (Python `-H 24`) | 12.3 s | 1.97 s | 6.2x |

In-process analysis of CIA.mp3 (excluding interpreter/process startup): Python 0.52 s
(shifts=1) / 1.27 s (shifts=4); C++ ~0.057 s / ~0.10 s. Loading a `.pklz` costs ~0.24 s
of gzip inflate for the default 400 MB table, which the native format avoids.

About 45-80 ms of every C++ run is process startup: the distro `libavcodec-freeworld` pulls in
236 shared libraries (x265, vvenc, ...) and runs their constructors. A static minimal FFmpeg
build would remove that.
