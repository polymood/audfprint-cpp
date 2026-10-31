#!/bin/bash
# Compare this C++ audfprint against the Python reference, end to end.
#
#   tests/parity.sh <path/to/python/audfprint> <long-audio-file> [python]
#
# Needs ffmpeg on PATH and a python with numpy, scipy, docopt, joblib, psutil.
# Builds a few tracks/queries from the audio file in a temp dir, then checks
# that precompute files are byte-identical and that match/list output agrees
# across options, database operations and .pklz files written by either side.
set -u
PYDIR=$(cd "$1" && pwd); SRC=$(realpath "$2"); PYTHON=${3:-python3}
CPP=$(cd "$(dirname "$0")/.." && pwd)/audfprint
PY="$PYTHON -W ignore $PYDIR/audfprint.py"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; cd "$W"
f() { ffmpeg -v error -y "$@"; }
for i in 0 1 2 3; do f -ss $((i * 30)) -t 30 -i "$SRC" -c:a libmp3lame -b:a 192k t$i.mp3; done
f -ss 15 -t 40 -i "$SRC" -af "asetrate=46000,aresample=48000" t4.wav
f -ss 50 -t 30 -i "$SRC" -ar 22050 -ac 1 t5.flac
f -ss 40 -t 12 -i "$SRC" -af "volume=0.3,highpass=f=200" -c:a libmp3lame -b:a 64k q1.mp3
f -ss 62 -t 8 -i "$SRC" -ar 8000 q2.wav
f -f lavfi -i "anoisesrc=d=6:c=pink:a=0.1" -ar 11025 q3.wav
T="t0.mp3 t1.mp3 t2.mp3 t3.mp3 t4.wav t5.flac"; Q="q1.mp3 q2.wav q3.wav"
m() { grep -E "^(Matched|NOMATCH)"; }
l() { grep -E "hashes\)$"; }
fail=0
check() {
  if [ "$2" == "$3" ] && [ -n "$2" ]; then echo "SAME  $1"; else fail=1; echo "DIFF  $1"; diff <(echo "$2") <(echo "$3") | head; fi
}
mkdir p c
$PY precompute -p p/h $T >/dev/null 2>&1; $CPP precompute -p c/h $T >/dev/null
$PY precompute -K -p p/k $T >/dev/null 2>&1; $CPP precompute -K -p c/k $T >/dev/null
$PY precompute -i 4 -p p/q q1.mp3 >/dev/null 2>&1; $CPP precompute -i 4 -p c/q q1.mp3 >/dev/null
check "precompute .afpt/.afpk byte-identical" "$(cd p && find . -type f | sort | xargs md5sum)" "$(cd c && find . -type f | sort | xargs md5sum)"
$PY new -d p/a.pklz t0.mp3 t1.mp3 t2.mp3 >/dev/null 2>&1; $CPP new -d c/a.pklz t0.mp3 t1.mp3 t2.mp3 >/dev/null
$PY add -d p/a.pklz t3.mp3 t4.wav t5.flac >/dev/null 2>&1; $CPP add -d c/a.pklz t3.mp3 t4.wav t5.flac >/dev/null
for o in "" "-X" "-R" "-x 3 -R" "-x 3 -X -R" "-i 1" "-w 1 -N 10" "-T -x 4" "-D 1"; do
  check "match [$o]" "$($PY match $o -d p/a.pklz $Q 2>/dev/null | m)" "$($CPP match $o -d c/a.pklz $Q | m)"
done
check "python reads C++ .pklz" "$($PY match -x 3 -R -d p/a.pklz $Q 2>/dev/null | m)" "$($PY match -x 3 -R -d c/a.pklz $Q 2>/dev/null | m)"
check "C++ reads python .pklz" "$($PY match -x 3 -R -d p/a.pklz $Q 2>/dev/null | m)" "$($CPP match -x 3 -R -d p/a.pklz $Q | m)"
check "list" "$($PY list -d p/a.pklz 2>/dev/null | l)" "$($CPP list -d c/a.pklz | l)"
for o in "-n 40 -F 5" "-S 20 -P 3" "-h 16" "-u 16" "-r 8000" "-b 10"; do
  $PY new $o -d p/o.pklz $T >/dev/null 2>&1; $CPP new $o -d c/o.db $T >/dev/null
  check "new+match [$o]" "$($PY match $o -x 2 -R -d p/o.pklz $Q 2>/dev/null | m)" "$($CPP match $o -x 2 -R -d c/o.db $Q | m)"
done
$PY new -d p/h.pklz p/h/t*.afpt >/dev/null 2>&1; $CPP new -d c/h.db c/h/t*.afpt >/dev/null
check "precomputed .afpt db and query" "$($PY match -d p/h.pklz p/q/q1.afpt 2>/dev/null | m | sed 's#p/##g')" "$($CPP match -d c/h.db c/q/q1.afpt | m | sed 's#c/##g')"
$PY new -d p/k.pklz p/k/t0.afpk p/k/t4.afpk >/dev/null 2>&1; $CPP new -d c/k.db c/k/t0.afpk c/k/t4.afpk >/dev/null
check "precomputed .afpk db" "$($PY match -x 2 -d p/k.pklz $Q 2>/dev/null | m)" "$($CPP match -x 2 -d c/k.db $Q | m)"
$PY new -d p/m1.pklz t0.mp3 t1.mp3 >/dev/null 2>&1; $CPP new -d c/m1.pklz t0.mp3 t1.mp3 >/dev/null
$PY new -d p/m2.pklz t2.mp3 t4.wav >/dev/null 2>&1; $CPP new -d c/m2.pklz t2.mp3 t4.wav >/dev/null
$PY newmerge -d p/m3.pklz p/m1.pklz p/m2.pklz >/dev/null 2>&1; $CPP newmerge -d c/m3.pklz c/m1.pklz c/m2.pklz >/dev/null
check "newmerge" "$($PY match -x 2 -d p/m3.pklz $Q 2>/dev/null | m)" "$($CPP match -x 2 -d c/m3.pklz $Q | m)"
$PY merge -d p/m1.pklz p/m2.pklz >/dev/null 2>&1; $CPP merge -d c/m1.pklz c/m2.pklz >/dev/null
check "merge" "$($PY match -x 2 -d p/m1.pklz $Q 2>/dev/null | m)" "$($CPP match -x 2 -d c/m1.pklz $Q | m)"
$PY remove -d p/a.pklz t1.mp3 t4.wav >/dev/null 2>&1; $CPP remove -d c/a.pklz t1.mp3 t4.wav >/dev/null
check "remove: list" "$($PY list -d p/a.pklz 2>/dev/null | l)" "$($CPP list -d c/a.pklz | l)"
check "remove: match" "$($PY match -x 3 -d p/a.pklz $Q 2>/dev/null | m)" "$($CPP match -x 3 -d c/a.pklz $Q | m)"
$PY add -d p/a.pklz t4.wav >/dev/null 2>&1; $CPP add -d c/a.pklz t4.wav >/dev/null
check "re-add: match" "$($PY match -x 3 -R -X -d p/a.pklz $Q 2>/dev/null | m)" "$($CPP match -x 3 -R -X -d c/a.pklz $Q | m)"
$PYTHON -W ignore - "$PYDIR" <<'EOF'
import sys, gzip, pickle
sys.path.insert(0, sys.argv[1])
import hash_table  # noqa: F401  (pickle needs the class)
ht = pickle.load(gzip.open('p/m3.pklz'), encoding='latin1')
pickle.dump(ht, gzip.open('p/proto2.pklz', 'wb'), protocol=2)
ht.table[ht.table != 0] -= 1 << ht.maxtimebits  # pre-2017 layout: ids start at 0
ht.maxtime = 1 << ht.maxtimebits; del ht.maxtimebits; ht.ht_version = 20140920
pickle.dump(ht, gzip.open('p/old.pklz', 'wb'), protocol=2)
EOF
check "protocol-2 pickle" "$($PY match -x 2 -d p/m3.pklz $Q 2>/dev/null | m)" "$($CPP match -x 2 -d p/proto2.pklz $Q | m)"
check "old-version pickle" "$($PY match -x 2 -d p/old.pklz $Q 2>/dev/null | m)" "$($CPP match -x 2 -d p/old.pklz $Q | m)"
[ $fail == 0 ] && echo "ALL SAME" || echo "SOME DIFFERENCES"
exit $fail
