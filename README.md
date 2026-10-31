# audfprint-cpp

A C++ reimplementation of [audfprint](https://github.com/dpwe/audfprint).

## Build

```
make
```

Needs g++ (C++20), zlib and the FFmpeg 8.1.3 shared libraries. FFmpeg headers and
pocketfft are included in `third_party/`.

## Usage

Same commands and options as the Python version (`./audfprint --help`):

```
./audfprint new   --dbase fpdbase.pklz tracks/*.mp3
./audfprint add   --dbase fpdbase.pklz more/*.mp3
./audfprint match --dbase fpdbase.pklz query.mp3
```

`.pklz` databases are compatible with the Python version. Not implemented: Matlab `.mat`
databases and the `--illustrate` plots.

`tests/parity.sh <python-audfprint-dir> <audio-file>` compares the output of both versions.

## Benchmark

4.4 minute MP3, Ryzen 9 5900X, wall time per command.

| command | Python | C++ |
|---|---|---|
| `precompute` | 1.18 s | 0.11 s |
| `new` | 2.61 s | 0.21 s |
| `match` (15 s query) | 2.32 s | 0.31 s |
| `new`, 32 files (Python `-H 24`) | 29.2 s | 1.31 s |

## License

MIT, like the original. FFmpeg headers are LGPL 2.1, pocketfft is BSD-3.
