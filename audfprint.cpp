// audfprint.cpp - C++ port of dpwe/audfprint (landmark-based audio fingerprinting).
// Algorithm follows audfprint_analyze.py / hash_table.py / audfprint_match.py
// step for step; speed comes from in-process FFmpeg decoding (MP3 decoded in
// parallel chunks), pocketfft, AVX2 log, and threads over files and shifts.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/intreadwrite.h>
}
#include <zlib.h>
extern "C" {
}
#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <memory>
#include <tuple>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "third_party/pocketfft_hdronly.h"

extern "C" __m256d _ZGVdN4v_log(__m256d);  // glibc libmvec AVX2 log

namespace fs = std::filesystem;
using std::string;
using std::vector;

// ---------------------------------------------------------------- utilities

// Persistent worker pool. parallel_for nests freely: the caller works through
// its own job while idle workers help, so nested calls cannot deadlock, and
// thread-local scratch buffers survive between calls.
class Pool {
    struct Job {
        const std::function<void(int)> *fn;
        int n;
        std::atomic<int> next{0}, active{0};
    };
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::deque<Job *> jobs;
    vector<std::thread> ws;
    bool stop = false;
    static void drain(Job *j) {
        for (int i; (i = j->next.fetch_add(1)) < j->n;) (*j->fn)(i);
    }

  public:
    explicit Pool(int n) {
        for (int t = 0; t < n - 1; t++)
            ws.emplace_back([this] {
                std::unique_lock lk(mu);
                for (;;) {
                    cv.wait(lk, [&] { return stop || !jobs.empty(); });
                    if (stop) return;
                    Job *j = jobs.front();
                    if (j->next.load() >= j->n) {
                        jobs.pop_front();
                        continue;
                    }
                    j->active++;
                    lk.unlock();
                    drain(j);
                    lk.lock();
                    if (--j->active == 0) done_cv.notify_all();  // owner frees j only after this
                }
            });
    }
    ~Pool() {
        {
            std::lock_guard lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto &t : ws) t.join();
    }
    int size() const { return (int)ws.size() + 1; }
    void run(int n, const std::function<void(int)> &fn) {
        Job j{&fn, n};
        {
            std::lock_guard lk(mu);
            jobs.push_back(&j);
        }
        cv.notify_all();
        drain(&j);
        std::unique_lock lk(mu);
        auto it = std::find(jobs.begin(), jobs.end(), &j);
        if (it != jobs.end()) jobs.erase(it);
        done_cv.wait(lk, [&] { return j.active == 0; });
    }
};

static int g_threads = 0;  // pool size, set from --ncores before first use
static Pool &pool() {
    static Pool p(g_threads > 0 ? g_threads : std::max(1u, std::thread::hardware_concurrency()));
    return p;
}

// nthreads <= 1 runs inline (used when the outer level already fills the cores)
static void parallel_for(int n, int nthreads, const std::function<void(int)> &fn) {
    if (n <= 0) return;
    if (n == 1 || nthreads <= 1 || pool().size() == 1) {
        for (int i = 0; i < n; i++) fn(i);
        return;
    }
    pool().run(n, fn);
}

// Allocator that backs big buffers with transparent huge pages (fewer page faults).
template <class T> struct HugeAlloc {
    using value_type = T;
    HugeAlloc() = default;
    template <class U> HugeAlloc(const HugeAlloc<U> &) {}
    T *allocate(size_t n) {
        const size_t HP = 2u << 20;
        size_t bytes = n * sizeof(T);
        void *p = bytes < 2 * HP ? malloc(bytes) : aligned_alloc(HP, (bytes + HP - 1) / HP * HP);
        if (!p) throw std::bad_alloc();
        if (bytes >= 2 * HP) madvise(p, (bytes + HP - 1) / HP * HP, MADV_HUGEPAGE);
        return (T *)p;
    }
    void deallocate(T *p, size_t) { free(p); }
    template <class U> bool operator==(const HugeAlloc<U> &) const { return true; }
};
template <class T> using hvec = vector<T, HugeAlloc<T>>;

static string now_ctime() {
    time_t t = time(nullptr);
    string s = ctime(&t);
    if (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
}

static string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
static string fmt(const char *f, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

static double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- decoding

// Equivalent of `ffmpeg -i file -f s16le -ac 1 -ar <sr> -` scaled to [-1, 1).
static hvec<float> audio_read(const string &path, int sr, int nthreads) {
    AVFormatContext *fc = nullptr;
    if (avformat_open_input(&fc, path.c_str(), nullptr, nullptr) < 0)
        throw std::runtime_error("cannot open " + path);
    std::unique_ptr<AVFormatContext, void (*)(AVFormatContext *)> fguard(
        fc, [](AVFormatContext *f) { avformat_close_input(&f); });
    int si = av_find_best_stream(fc, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (si < 0 || fc->streams[si]->codecpar->sample_rate <= 0 ||
        fc->streams[si]->codecpar->ch_layout.nb_channels <= 0) {
        avformat_find_stream_info(fc, nullptr);
        si = av_find_best_stream(fc, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    }
    if (si < 0) throw std::runtime_error("no audio stream in " + path);
    AVStream *st = fc->streams[si];
    const AVCodecParameters *par = st->codecpar;
    const AVCodec *codec = avcodec_find_decoder(par->codec_id);
    if (!codec) throw std::runtime_error("no decoder for " + path);

    // Parallel MP3 decode + resample, overlapped with demuxing. Every MP3
    // packet decodes to frame_size samples (minus the encoder delay on packet
    // 0), so the global sample index of any packet is known up front. Chunk c
    // decodes its packets starting WARM packets early (decoder state - bit
    // reservoir, MDCT overlap, synthesis window - is fully refreshed by then),
    // plus resampler history and lookahead. swresample with an exact rational
    // ratio si:so computes each output from a fixed filter phase, so a fresh
    // context fed from a multiple of si reproduces the sequential output
    // exactly. Frame sizes are checked; any surprise falls back to one
    // sequential pass, which is also the path for all other codecs.
    const long FS = par->frame_size, in_rate = par->sample_rate;
    const long g = in_rate > 0 ? std::gcd(in_rate, (long)sr) : 1, si_ = std::max(1L, in_rate / g), so_ = sr / g;
    const long PRE = (2048 / si_ + 1) * si_, WARM = 4;
    const bool parallel = par->codec_id == AV_CODEC_ID_MP3 && FS > 0 && in_rate > 0 && so_ <= 1024 && nthreads > 1;

    // Packet store: filled by the demux task while decode tasks consume it.
    constexpr long BS = 4096;
    vector<std::unique_ptr<AVPacket *[]>> blocks(1 << 16);
    std::mutex dm;
    std::condition_variable dcv;
    long avail = 0;  // guarded by dm
    bool eof = false;
    long skip = 0;  // encoder delay, from packet 0 side data
    auto pkt = [&](long i) { return blocks[i / BS][i % BS]; };
    std::unique_ptr<void, std::function<void(void *)>> pguard((void *)1, [&](void *) {
        for (long i = 0; i < avail; i++) {
            AVPacket *p = pkt(i);
            av_packet_free(&p);
        }
    });
    auto demux = [&] {
        long n = 0;
        for (AVPacket *p = av_packet_alloc();;) {
            if (av_read_frame(fc, p) < 0 || n / BS >= (long)blocks.size()) {
                av_packet_free(&p);
                break;
            }
            if (p->stream_index != si) {
                av_packet_unref(p);
                continue;
            }
            if (n == 0) {
                const AVPacketSideData *sd = av_packet_side_data_get(p->side_data, p->side_data_elems, AV_PKT_DATA_SKIP_SAMPLES);
                if (sd && sd->size >= 4) skip = AV_RL32(sd->data);
            }
            if (!blocks[n / BS]) blocks[n / BS].reset(new AVPacket *[BS]);
            p->opaque = (void *)(intptr_t)(n + 1);
            blocks[n / BS][n % BS] = p;
            p = av_packet_alloc();
            if (++n % 64 == 0) {
                std::lock_guard lk(dm);
                avail = n;
                dcv.notify_all();
            }
        }
        std::lock_guard lk(dm);
        avail = n, eof = true;
        dcv.notify_all();
    };
    auto wait_for = [&](long need, bool &at_eof) {  // packets available once >= need, or EOF
        std::unique_lock lk(dm);
        dcv.wait(lk, [&] { return eof || avail >= need; });
        at_eof = eof;
        return avail;
    };
    auto start_of = [&](long i) { return std::max(0L, i * FS - skip); };  // first global sample of packet i

    std::atomic<bool> bad{false}, fatal{false};
    // Decode packets [s, pe), dropping output of packets < valid, and feed
    // global samples [F, G) to a fresh resampler. multi: check MP3 layout.
    auto decode = [&](long s, long valid, long pe, long npk, long F, long G, bool multi, bool flush, hvec<int16_t> &o) {
        AVCodecContext *ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(ctx, par);
        ctx->pkt_timebase = st->time_base;
        ctx->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
        ctx->thread_count = 1;
        if (avcodec_open2(ctx, codec, nullptr) < 0) {
            fatal = true;
            avcodec_free_context(&ctx);
            return;
        }
        SwrContext *swr = nullptr;
        int fmt0 = -1;
        size_t bps = 0, pos = 0;
        long fed_to = F;
        {  // size the output once: growing it by reallocation is slow with many threads
            double secs = multi ? (double)(pe - s) * FS / in_rate
                                : fc->duration > 0 ? (double)fc->duration / AV_TIME_BASE : 0.0;
            o.resize((size_t)(secs * sr * 1.01) + 8192);
        }
        auto convert = [&](const uint8_t **in, int n) {
            const int cap = (int)(n * so_ / si_) + 512;  // small windows: swr cost scales with out_count
            if (o.size() - pos < (size_t)cap) o.resize(pos + cap + o.size() / 2);
            uint8_t *op = (uint8_t *)(o.data() + pos);
            int got = swr_convert(swr, &op, cap, in, n);
            if (got > 0) pos += got;
            return got;
        };
        AVFrame *f = av_frame_alloc();
        auto drain = [&] {
            while (avcodec_receive_frame(ctx, f) == 0) {
                const long i = (long)(intptr_t)f->opaque - 1;
                if (i >= valid && !swr) {
                    fmt0 = f->format;
                    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
                    swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_S16, sr, &f->ch_layout,
                                        (AVSampleFormat)f->format, f->sample_rate, 0, nullptr);
                    if (!swr || swr_init(swr) < 0) fatal = true;
                    bps = av_get_bytes_per_sample((AVSampleFormat)f->format) *
                          (av_sample_fmt_is_planar((AVSampleFormat)f->format) ? 1 : f->ch_layout.nb_channels);
                }
                // ponytail: mid-stream format changes are dropped, not re-resampled
                if (i >= valid && !fatal && f->format == fmt0) {
                    long lo = 0, hi = f->nb_samples;  // part of this frame to feed
                    if (multi) {
                        const long fs = start_of(i);
                        if ((i + 1 < npk && f->nb_samples != start_of(i + 1) - fs) || f->sample_rate != in_rate) bad = true;
                        lo = std::clamp(F - fs, 0L, (long)f->nb_samples);
                        hi = std::clamp(G - fs, 0L, (long)f->nb_samples);
                        if (lo < hi) {
                            if (fs + lo != fed_to) bad = true;  // must feed one contiguous range from F
                            fed_to = fs + hi;
                        }
                    }
                    const int planes = av_sample_fmt_is_planar((AVSampleFormat)fmt0) ? std::min(f->ch_layout.nb_channels, 64) : 1;
                    for (long off = lo; off < hi; off += 8192) {
                        const uint8_t *in[64] = {};
                        for (int p = 0; p < planes; p++) in[p] = f->extended_data[p] + off * bps;
                        convert(in, (int)std::min(8192L, hi - off));
                    }
                }
                av_frame_unref(f);
            }
        };
        for (long i = s; i < pe && !bad && !fatal; i++) {
            avcodec_send_packet(ctx, pkt(i));  // decode errors ignored, like the ffmpeg CLI
            drain();
        }
        avcodec_send_packet(ctx, nullptr);
        drain();
        av_frame_free(&f);
        avcodec_free_context(&ctx);
        if (swr && flush)
            while (convert(nullptr, 0) > 0) {}
        swr_free(&swr);
        if (multi && !flush && fed_to != G) bad = true;
        o.resize(pos);
    };
    auto to_float = [&](vector<hvec<int16_t>> &seg) {
        vector<size_t> at(seg.size() + 1, 0);
        for (size_t k = 0; k < seg.size(); k++) at[k + 1] = at[k] + seg[k].size();
        hvec<float> out(at.back());
        parallel_for((int)seg.size(), nthreads, [&](int k) {
            for (size_t i = 0; i < seg[k].size(); i++) out[at[k] + i] = seg[k][i] * (1.0f / 32768.0f);
        });
        return out;
    };

    if (parallel) {
        // chunks of K packets, ~2 per thread, sized from the container's duration estimate
        const long est = fc->duration > 0 ? (long)((double)fc->duration / AV_TIME_BASE * in_rate / FS) + 1 : 1 << 14;
        const long K = std::max(256L, est / (2L * nthreads)), NCH = est / K + 2, LOOK = PRE / FS + 3;
        vector<hvec<int16_t>> seg(NCH);
        parallel_for((int)NCH + 1, nthreads, [&](int t) {
            if (t == 0) return demux();
            const long c = t - 1, b = c * K;
            bool at_eof;
            // Chunk c is last iff the stream ends within LOOK packets of its end
            // (it then absorbs the tail); chunks after the last one are empty.
            // Waiting for one packet more than needed makes the call exact.
            long npk = wait_for(c == NCH - 1 ? LONG_MAX : (c + 1) * K + LOOK + 1, at_eof);
            if (c > 0 && at_eof && b + LOOK >= npk) return;
            const bool last = c == NCH - 1 || (at_eof && (c + 1) * K + LOOK >= npk);
            if (!at_eof) npk = LONG_MAX;  // last packet not seen yet, so none of ours is it
            auto A = [&](long cc) { return cc == 0 ? 0L : start_of(cc * K) / si_ * si_; };
            const long a0 = A(c), a1 = last ? LONG_MAX : A(c + 1);
            const long F = c == 0 ? 0 : a0 - PRE, G = last ? LONG_MAX : a1 + PRE;
            long s = 0, valid = 0, pe = std::min(npk, (c + 1) * K + LOOK);
            if (last) pe = npk;
            while (s < b && start_of(s + 1) <= F) s++;  // first packet holding sample F
            if (s > 0) s = std::max(0L, s - WARM), valid = s == 0 ? 0 : s + WARM;
            hvec<int16_t> &o = seg[c];
            decode(s, valid, pe, npk, F, G, true, last, o);
            const long drop = (a0 - F) / si_ * so_, keep = last ? (long)o.size() - drop : (a1 - a0) / si_ * so_;
            if (bad || keep < 0 || drop + keep > (long)o.size()) {
                bad = true;
                return;
            }
            o.erase(o.begin(), o.begin() + drop);
            o.resize(keep);
        });
        if (fatal) throw std::runtime_error("cannot decode " + path);
        if (!bad) return to_float(seg);
        bad = false;  // layout assumption broken: redo sequentially (packets are all read)
    } else {
        demux();
    }
    vector<hvec<int16_t>> seg(1);
    decode(0, 0, avail, avail, 0, LONG_MAX, false, true, seg[0]);
    if (fatal) throw std::runtime_error("cannot decode " + path);
    return to_float(seg);
}

// ---------------------------------------------------------------- analysis

constexpr int N_FFT = 512, N_HOP = 256, NBINS = 256;  // Nyquist bin dropped, as in Python
constexpr int F1_BITS = 8, DF_BITS = 6, DT_BITS = 6;
constexpr uint32_t B1_MASK = (1 << F1_BITS) - 1, DF_MASK = (1 << DF_BITS) - 1,
                   DT_MASK = (1 << DT_BITS) - 1;
constexpr int B1_SHIFT = DF_BITS + DT_BITS, DF_SHIFT = DT_BITS;

struct Peak { int32_t t, bin; };

// Reusable 2 MB-aligned buffer with transparent huge pages (fewer page faults).
struct BigBuf {
    double *p = nullptr;
    size_t cap = 0;
    ~BigBuf() { free(p); }
    double *get(size_t n) {
        if (n > cap) {
            free(p);
            const size_t HP = 2u << 20, bytes = (n * sizeof(double) + HP - 1) / HP * HP;
            p = (double *)aligned_alloc(HP, bytes);
            if (!p) throw std::bad_alloc();
            madvise(p, bytes, MADV_HUGEPAGE);
            cap = bytes / sizeof(double);
        }
        return p;
    }
};
struct TH { int32_t t, h; };  // (time, hash)

struct Analyzer {
    double density = 20.0, f_sd = 30.0;
    int maxpksperframe = 5, maxpairsperpeak = 3, shifts = 1, target_sr = 11025;
    int mindt = 2, targetdt = 63, targetdf = 31;
    bool fail_on_error = true;

    double win[N_FFT];
    double gauss[2 * NBINS + 1];  // exp(-0.5*((k/f_sd)^2)), k = -256..256

    void init() {
        for (int i = 0; i < N_FFT; i++) {  // np.hanning(514)[1:-1]
            double n = 1 - 514 + 2 * (i + 1);
            win[i] = 0.5 + 0.5 * std::cos(M_PI * n / 513.0);
        }
        for (int k = -NBINS; k <= NBINS; k++) {
            double x = k / f_sd;
            gauss[k + NBINS] = std::exp(-0.5 * (x * x));
        }
    }

    // threshold = max(threshold, val * gaussian centred on pos)
    void spread(double *th, int pos, double val) const {
        const double *g = gauss + NBINS - pos;
        const __m256d v = _mm256_set1_pd(val);
        for (int i = 0; i < NBINS; i += 4)
            _mm256_storeu_pd(th + i, _mm256_max_pd(_mm256_loadu_pd(th + i), _mm256_mul_pd(v, _mm256_loadu_pd(g + i))));
    }

    // locmax: v[i] >= v[i-1] (or i == 0) and v[i] > v[i+1] (or last)
    static bool is_locmax(const double *v, int i, int n) {
        return (i == 0 || v[i] >= v[i - 1]) && !(i + 1 < n && v[i + 1] >= v[i]);
    }

    void spread_locmaxes(const double *v, double *th) const {
        std::fill(th, th + NBINS, 0.0);
        for (int i = 0; i < NBINS; i++)
            if (is_locmax(v, i, NBINS)) spread(th, i, v[i]);
    }

    vector<Peak> find_peaks(const float *d, size_t n, int nthreads) const {
        vector<Peak> pk;
        if (n == 0) return pk;
        const double a_dec = 1.0 - 0.01 * (density * std::sqrt(N_HOP / 352.8) / 35.0);
        const size_t ncols = 1 + n / N_HOP;
        // Spectrogram rows: [-inf, bin0..bin255, -inf, pad]; sentinels make the
        // locmax test branch-free at the edges with identical semantics.
        constexpr int ST = NBINS + 8, OFF = 4;
        thread_local BigBuf store;
        double *sg = store.get(ncols * ST);
        vector<double> nyq(ncols);
        // reflect-padded sample access (np.pad mode='reflect', any length)
        auto padded = [&](long x) -> double {
            x -= N_FFT / 2;
            if (n == 1) return d[0];
            long period = 2 * (long)(n - 1), m = x % period;
            if (m < 0) m += period;
            if (m >= (long)n) m = period - m;
            return d[m];
        };
        // 1) log|STFT| in blocks of frames, batched FFTs (SIMD across frames)
        constexpr int BLK = 64;
        const int nblk = (int)((ncols + BLK - 1) / BLK);
        vector<double> bmax(nblk, 0.0), bsum(nblk, 0.0);
        parallel_for(nblk, nthreads, [&](int bi) {
            thread_local vector<double> in(BLK * N_FFT);
            thread_local vector<std::complex<double>> out(BLK * (N_FFT / 2 + 1));
            const size_t c0 = (size_t)bi * BLK, nb = std::min<size_t>(BLK, ncols - c0);
            for (size_t j = 0; j < nb; j++) {
                long base = (long)((c0 + j) * N_HOP);
                double *w = &in[j * N_FFT];
                if (base >= N_FFT / 2 && base + N_FFT / 2 < (long)n) {
                    const float *s = d + base - N_FFT / 2;
                    for (int i = 0; i < N_FFT; i++) w[i] = (double)s[i] * win[i];
                } else {
                    for (int i = 0; i < N_FFT; i++) w[i] = padded(base + i) * win[i];
                }
            }
            pocketfft::r2c<double>({nb, (size_t)N_FFT}, {N_FFT * sizeof(double), sizeof(double)},
                                   {(N_FFT / 2 + 1) * sizeof(std::complex<double>), sizeof(std::complex<double>)},
                                   1, true, in.data(), out.data(), 1.0, 1);
            double m = 0.0;
            for (size_t j = 0; j < nb; j++) {
                const std::complex<double> *o = &out[j * (N_FFT / 2 + 1)];
                double *row = sg + (c0 + j) * ST;
                row[OFF - 1] = row[OFF + NBINS] = -INFINITY;
                for (int k = 0; k < NBINS; k++) {
                    double re = o[k].real(), im = o[k].imag();
                    row[OFF + k] = std::sqrt(re * re + im * im);
                }
                nyq[c0 + j] = std::fabs(o[NBINS].real());
                double rm = nyq[c0 + j];
                for (int k = 0; k < NBINS; k++) rm = std::max(rm, row[OFF + k]);
                m = std::max(m, rm);
                for (int k = 0; k < NBINS; k += 4)  // log now, floor later (log is monotonic)
                    _mm256_storeu_pd(row + OFF + k, _ZGVdN4v_log(_mm256_loadu_pd(row + OFF + k)));
            }
            bmax[bi] = m;
        });
        const double gmax = *std::max_element(bmax.begin(), bmax.end());
        // 2) log(max(sg, max/1e6)) and its mean (which includes the Nyquist row)
        double mean = 0.0;
        if (gmax > 0.0) {
            const double floor_ = gmax / 1e6;
            double lf4[4];
            _mm256_storeu_pd(lf4, _ZGVdN4v_log(_mm256_set1_pd(floor_)));
            const __m256d lf = _mm256_set1_pd(lf4[0]);
            parallel_for(nblk, nthreads, [&](int bi) {
                const size_t c0 = (size_t)bi * BLK, c1 = std::min<size_t>(c0 + BLK, ncols);
                double s = 0.0;
                for (size_t c = c0; c < c1; c++) {
                    double *row = sg + c * ST + OFF;
                    __m256d acc = _mm256_setzero_pd();
                    for (int k = 0; k < NBINS; k += 4) {
                        __m256d v = _mm256_max_pd(_mm256_loadu_pd(row + k), lf);
                        _mm256_storeu_pd(row + k, v);
                        acc = _mm256_add_pd(acc, v);
                    }
                    double a[4];
                    _mm256_storeu_pd(a, acc);
                    s += ((a[0] + a[1]) + (a[2] + a[3])) + std::log(std::max(nyq[c], floor_));
                }
                bsum[bi] = s;
            });
            double total = 0.0;
            for (double s : bsum) total += s;
            mean = total / ((double)ncols * (NBINS + 1));
        } else {
            fprintf(stderr, "find_peaks: Warning: input signal is identically zero.\n");
            for (size_t c = 0; c < ncols; c++) std::fill(sg + c * ST + OFF, sg + c * ST + OFF + NBINS, 0.0);
        }
        // 3) (x - mean) then onset-emphasis HPF along time: y = x - x[-1] + 0.98 y[-1]
        constexpr int GRP = 32;
        parallel_for(NBINS / GRP, nthreads, [&](int g) {
            double z[GRP] = {0};
            for (size_t c = 0; c < ncols; c++) {
                double *row = sg + c * ST + OFF + g * GRP;
                for (int k = 0; k < GRP; k++) {
                    double x = row[k] - mean, y = x + z[k];
                    z[k] = -x + 0.98 * y;
                    row[k] = y;
                }
            }
        });
        // 4) forward pass: decaying threshold, keep up to maxpksperframe peaks/frame
        vector<std::array<uint64_t, 4>> bits(ncols, {0, 0, 0, 0});
        alignas(32) double th[NBINS];
        {
            double v0[NBINS];
            std::fill(v0, v0 + NBINS, -INFINITY);
            for (size_t c = 0; c < std::min<size_t>(10, ncols); c++)
                for (int k = 0; k < NBINS; k++) v0[k] = std::max(v0[k], sg[c * ST + OFF + k]);
            spread_locmaxes(v0, th);
        }
        const __m256d vdec = _mm256_set1_pd(a_dec);
        auto decay = [&] {
            for (int k = 0; k < NBINS; k += 4) _mm256_store_pd(th + k, _mm256_mul_pd(_mm256_load_pd(th + k), vdec));
        };
        const int maxpk = maxpksperframe;
        vector<std::pair<double, int>> top;
        top.reserve(maxpk + 1);
        for (size_t c = 0; c < ncols; c++) {
            const double *s = sg + c * ST + OFF;
            // candidates: s > th and local max (s >= left, !(right >= s))
            uint64_t cand[4] = {0, 0, 0, 0};
            for (int k = 0; k < NBINS; k += 4) {
                __m256d sv = _mm256_loadu_pd(s + k);
                __m256d m = _mm256_and_pd(_mm256_cmp_pd(sv, _mm256_load_pd(th + k), _CMP_GT_OQ),
                                          _mm256_cmp_pd(sv, _mm256_loadu_pd(s + k - 1), _CMP_GE_OQ));
                m = _mm256_andnot_pd(_mm256_cmp_pd(_mm256_loadu_pd(s + k + 1), sv, _CMP_GE_OQ), m);
                cand[k >> 6] |= (uint64_t)_mm256_movemask_pd(m) << (k & 63);
            }
            top.clear();
            for (int w = 0; w < 4; w++)
                for (uint64_t m = cand[w]; m; m &= m - 1) {
                    int k = w * 64 + __builtin_ctzll(m);
                    std::pair<double, int> e{s[k], k};
                    if ((int)top.size() == maxpk && !(e > top.back())) continue;
                    top.insert(std::upper_bound(top.begin(), top.end(), e, std::greater<>()), e);
                    if ((int)top.size() > maxpk) top.pop_back();
                }
            for (auto &[val, k] : top) {
                spread(th, k, val);
                bits[c][k >> 6] |= 1ull << (k & 63);
            }
            decay();
        }
        // 5) backward pass: prune peaks closely followed (earlier) by bigger ones
        spread_locmaxes(sg + (ncols - 1) * ST + OFF, th);
        vector<std::pair<double, int>> list;
        for (size_t col = ncols; col >= 1; col--) {
            size_t c = col - 1;
            const double *s = sg + c * ST + OFF;
            list.clear();
            for (int w = 0; w < 4; w++)
                for (uint64_t m = bits[c][w]; m; m &= m - 1) {
                    int k = w * 64 + __builtin_ctzll(m);
                    list.push_back({s[k], k});
                }
            std::sort(list.begin(), list.end(), std::greater<>());
            for (auto &[val, k] : list) {
                if (val >= th[k]) {
                    spread(th, k, val);
                    if (col < ncols) bits[col][k >> 6] &= ~(1ull << (k & 63));
                } else {
                    bits[c][k >> 6] &= ~(1ull << (k & 63));
                }
            }
            decay();
        }
        for (size_t c = 0; c < ncols; c++)
            for (int w = 0; w < 4; w++)
                for (uint64_t m = bits[c][w]; m; m &= m - 1)
                    pk.push_back({(int32_t)c, w * 64 + __builtin_ctzll(m)});
        return pk;
    }

    // peaks2landmarks + landmarks2hashes, appending (time<<32 | hash) keys
    void peaks2hashes(const vector<Peak> &pk, vector<uint64_t> &out) const {
        if (pk.empty()) return;
        const int scols = pk.back().t + 1;
        vector<int> start(scols + 1, 0);
        for (auto &p : pk) start[p.t + 1]++;
        for (int c = 0; c < scols; c++) start[c + 1] += start[c];
        for (int col = 0; col < scols; col++)
            for (int i = start[col]; i < start[col + 1]; i++) {
                const int peak = pk[i].bin;
                int pairs = 0;
                for (int col2 = col + mindt; col2 < std::min(scols, col + targetdt) && pairs < maxpairsperpeak; col2++)
                    for (int j = start[col2]; j < start[col2 + 1] && pairs < maxpairsperpeak; j++) {
                        const int peak2 = pk[j].bin;
                        if (std::abs(peak2 - peak) < targetdf) {
                            uint32_t h = ((peak & B1_MASK) << B1_SHIFT) |
                                         (((uint32_t)(peak2 - peak) & DF_MASK) << DF_SHIFT) |
                                         ((uint32_t)(col2 - col) & DT_MASK);
                            out.push_back(((uint64_t)(uint32_t)col << 32) | h);
                            pairs++;
                        }
                    }
            }
    }

    static vector<TH> unique_sorted(vector<uint64_t> &keys) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        vector<TH> r(keys.size());
        for (size_t i = 0; i < keys.size(); i++) r[i] = {(int32_t)(keys[i] >> 32), (int32_t)(uint32_t)keys[i]};
        return r;
    }

    // wavfile2peaks + wavfile2hashes. dur gets the soundfile duration.
    vector<TH> file2hashes(const string &fn, int nshifts, int nthreads, double &dur) const;
    vector<Peak> file2peaks(const string &fn, int nthreads, double &dur) const;
};

static const char HASH_MAGIC[] = "audfprinthashV00", PEAK_MAGIC[] = "audfprintpeakV00";

static vector<std::array<int32_t, 2>> load_pairs(const string &fn, const char *magic) {
    std::ifstream f(fn, std::ios::binary);
    char m[16];
    if (!f.read(m, 16) || memcmp(m, magic, 16) != 0)
        throw std::runtime_error(fn + " is not a " + (magic == HASH_MAGIC ? "hash" : "peak") + " file");
    vector<std::array<int32_t, 2>> v;
    std::array<int32_t, 2> p;
    while (f.read((char *)p.data(), 8)) v.push_back(p);  // little-endian host assumed
    return v;
}

template <class T> static void save_pairs(const string &fn, const char *magic, const vector<T> &v) {
    FILE *f = fopen(fn.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + fn);
    fwrite(magic, 1, 16, f);
    fwrite(v.data(), sizeof(T), v.size(), f);
    fclose(f);
}

static string ext_of(const string &fn) { return fs::path(fn).extension().string(); }

vector<Peak> Analyzer::file2peaks(const string &fn, int nthreads, double &dur) const {
    if (ext_of(fn) == ".afpk") {
        auto v = load_pairs(fn, PEAK_MAGIC);
        vector<Peak> pk;
        int32_t mx = 0;
        for (auto &p : v) pk.push_back({p[0], p[1]}), mx = std::max(mx, p[0]);
        dur = (double)mx * N_HOP / target_sr;
        return pk;
    }
    auto d = audio_read(fn, target_sr, nthreads);
    dur = (double)d.size() / target_sr;
    return find_peaks(d.data(), d.size(), nthreads);
}

vector<TH> Analyzer::file2hashes(const string &fn, int nshifts, int nthreads, double &dur) const {
    string ext = ext_of(fn);
    if (ext == ".afpt") {
        auto v = load_pairs(fn, HASH_MAGIC);
        vector<TH> r;
        int32_t mx = 0;
        for (auto &p : v) r.push_back({p[0], p[1]}), mx = std::max(mx, p[0]);
        dur = (double)mx * N_HOP / target_sr;
        return r;
    }
    vector<uint64_t> keys;
    if (ext == ".afpk" || nshifts < 2) {
        peaks2hashes(file2peaks(fn, nthreads, dur), keys);
    } else {
        auto d = audio_read(fn, target_sr, nthreads);
        dur = (double)d.size() / target_sr;
        vector<vector<uint64_t>> per(nshifts);
        const int inner = std::max(1, nthreads / nshifts);
        parallel_for(nshifts, nthreads, [&](int s) {
            size_t off = (size_t)((double)s / nshifts * N_HOP);
            if (off < d.size()) peaks2hashes(find_peaks(d.data() + off, d.size() - off, inner), per[s]);
        });
        for (auto &p : per) keys.insert(keys.end(), p.begin(), p.end());
    }
    return unique_sorted(keys);
}

// ---------------------------------------------------------------- Python pickle (.pklz)
// Minimal pickle VM (protocols 0-5) - enough to read audfprint's gzipped
// HashTable pickles as written by any Python/numpy version - plus a writer
// that emits a pickle Python audfprint loads (numpy 1.x and 2.x).

struct PyObj;
using PyRef = std::shared_ptr<PyObj>;
struct PyObj {
    enum Kind { None, Bool, Int, Float, Str, Bytes, Tuple, List, Dict, Global, Call } kind = None;
    int64_t i = 0;
    double f = 0;
    string s;               // Str (utf-8), Global ("module.name")
    char *data = nullptr;   // Bytes (malloc'd, can be adopted)
    size_t n = 0;
    vector<PyRef> items;    // Tuple, List; Dict as k0, v0, k1, v1, ...
    PyRef func, args, state;  // Call: func(*args), then BUILD state
    ~PyObj() { free(data); }
    PyRef get(const string &key) const {
        for (size_t k = 0; k + 1 < items.size(); k += 2)
            if (items[k]->kind == Str && items[k]->s == key) return items[k + 1];
        return nullptr;
    }
};

class Unpickler {
    gzFile f;
    vector<PyRef> st;
    vector<size_t> marks;
    std::unordered_map<int64_t, PyRef> memo;

    void rd(void *p, size_t n) {
        for (size_t got = 0; got < n;) {
            int r = gzread(f, (char *)p + got, (unsigned)std::min<size_t>(n - got, 1u << 30));
            if (r <= 0) throw std::runtime_error("truncated or corrupt pickle");
            got += r;
        }
    }
    template <class T> T rdv() {
        T v;
        rd(&v, sizeof v);
        return v;
    }
    string line() {
        string s;
        for (char c; (c = rdv<char>()) != '\n';) s += c;
        return s;
    }
    static PyRef mk(PyObj::Kind k, int64_t i = 0) {
        auto o = std::make_shared<PyObj>();
        o->kind = k, o->i = i;
        return o;
    }
    PyRef str(size_t n, PyObj::Kind k) {
        auto o = mk(k);
        if (k == PyObj::Bytes) {
            o->data = (char *)malloc(std::max<size_t>(n, 1));
            if (!o->data) throw std::bad_alloc();
            o->n = n;
            rd(o->data, n);
        } else {
            o->s.resize(n);
            rd(o->s.data(), n);
        }
        return o;
    }
    PyRef pop() {
        if (st.empty()) throw std::runtime_error("pickle stack underflow");
        PyRef o = st.back();
        st.pop_back();
        return o;
    }
    PyRef &top() {
        if (st.empty()) throw std::runtime_error("pickle stack underflow");
        return st.back();
    }
    vector<PyRef> pop_mark() {
        if (marks.empty() || marks.back() > st.size()) throw std::runtime_error("pickle mark error");
        size_t m = marks.back();
        marks.pop_back();
        vector<PyRef> v(st.begin() + m, st.end());
        st.resize(m);
        return v;
    }
    PyRef seq(PyObj::Kind k, vector<PyRef> v) {
        auto o = mk(k);
        o->items = std::move(v);
        return o;
    }
    PyRef call(PyRef func, PyRef args) {
        auto o = mk(PyObj::Call);
        o->func = func, o->args = args;
        return o;
    }

  public:
    explicit Unpickler(gzFile f) : f(f) {}
    PyRef load() {
        using K = PyObj::Kind;
        for (;;) {
            uint8_t op = rdv<uint8_t>();
            switch (op) {
            case 0x80: rdv<uint8_t>(); break;   // PROTO
            case 0x95: rdv<uint64_t>(); break;  // FRAME
            case '.': return pop();             // STOP
            case '(': marks.push_back(st.size()); break;
            case '0': pop(); break;
            case '1': pop_mark(); break;
            case '2': st.push_back(top()); break;
            case 'N': st.push_back(mk(K::None)); break;
            case 0x88: st.push_back(mk(K::Bool, 1)); break;
            case 0x89: st.push_back(mk(K::Bool, 0)); break;
            case 'I': {
                string l = line();
                st.push_back(l == "01" ? mk(K::Bool, 1) : l == "00" ? mk(K::Bool, 0) : mk(K::Int, std::stoll(l)));
                break;
            }
            case 'L': {
                string l = line();
                if (!l.empty() && l.back() == 'L') l.pop_back();
                st.push_back(mk(K::Int, std::stoll(l)));
                break;
            }
            case 'J': st.push_back(mk(K::Int, rdv<int32_t>())); break;
            case 'K': st.push_back(mk(K::Int, rdv<uint8_t>())); break;
            case 'M': st.push_back(mk(K::Int, rdv<uint16_t>())); break;
            case 0x8a: case 0x8b: {  // LONG1 / LONG4: little-endian two's complement
                size_t n = op == 0x8a ? rdv<uint8_t>() : (size_t)rdv<int32_t>();
                vector<uint8_t> b(n);
                rd(b.data(), n);
                if (n > 8) throw std::runtime_error("pickle integer too large");
                uint64_t v = 0;
                for (size_t k = 0; k < n; k++) v |= (uint64_t)b[k] << (8 * k);
                if (n && n < 8 && (b[n - 1] & 0x80)) v |= ~0ull << (8 * n);
                st.push_back(mk(K::Int, (int64_t)v));
                break;
            }
            case 'G': {
                uint64_t v = __builtin_bswap64(rdv<uint64_t>());
                auto o = mk(K::Float);
                memcpy(&o->f, &v, 8);
                st.push_back(o);
                break;
            }
            case 'F': {
                auto o = mk(K::Float);
                o->f = std::stod(line());
                st.push_back(o);
                break;
            }
            case 'S': {  // STRING: quoted repr (no escape handling needed for our fields)
                string l = line();
                auto o = mk(K::Bytes);
                string body = l.size() >= 2 ? l.substr(1, l.size() - 2) : l;
                o->n = body.size();
                o->data = (char *)malloc(std::max<size_t>(o->n, 1));
                memcpy(o->data, body.data(), o->n);
                st.push_back(o);
                break;
            }
            case 'T': st.push_back(str((uint32_t)rdv<int32_t>(), K::Bytes)); break;
            case 'U': st.push_back(str(rdv<uint8_t>(), K::Bytes)); break;
            case 'B': st.push_back(str(rdv<uint32_t>(), K::Bytes)); break;
            case 'C': st.push_back(str(rdv<uint8_t>(), K::Bytes)); break;
            case 0x8e: case 0x96: st.push_back(str(rdv<uint64_t>(), K::Bytes)); break;  // BINBYTES8, BYTEARRAY8
            case 'X': st.push_back(str(rdv<uint32_t>(), K::Str)); break;
            case 0x8c: st.push_back(str(rdv<uint8_t>(), K::Str)); break;
            case 0x8d: st.push_back(str(rdv<uint64_t>(), K::Str)); break;
            case 'V': {
                auto o = mk(K::Str);
                o->s = line();
                st.push_back(o);
                break;
            }
            case ')': st.push_back(mk(K::Tuple)); break;
            case 't': st.push_back(seq(K::Tuple, pop_mark())); break;
            case 0x85: case 0x86: case 0x87: {
                vector<PyRef> v(op - 0x84);
                for (size_t k = v.size(); k-- > 0;) v[k] = pop();
                st.push_back(seq(K::Tuple, v));
                break;
            }
            case ']': case 0x8f: st.push_back(mk(K::List)); break;  // EMPTY_LIST, EMPTY_SET
            case 'l': case 0x91: st.push_back(seq(K::List, pop_mark())); break;  // LIST, FROZENSET
            case 'a': {
                PyRef v = pop();
                top()->items.push_back(v);
                break;
            }
            case 'e': case 0x90: {  // APPENDS, ADDITEMS
                auto v = pop_mark();
                auto &it = top()->items;
                it.insert(it.end(), v.begin(), v.end());
                break;
            }
            case '}': st.push_back(mk(K::Dict)); break;
            case 'd': st.push_back(seq(K::Dict, pop_mark())); break;
            case 's': {
                PyRef v = pop(), k = pop();
                top()->items.push_back(k);
                top()->items.push_back(v);
                break;
            }
            case 'u': {
                auto v = pop_mark();
                auto &it = top()->items;
                it.insert(it.end(), v.begin(), v.end());
                break;
            }
            case 'c': {
                auto o = mk(K::Global);
                string m = line();
                o->s = m + "." + line();
                st.push_back(o);
                break;
            }
            case 0x93: {
                PyRef name = pop(), mod = pop();
                auto o = mk(K::Global);
                o->s = mod->s + "." + name->s;
                st.push_back(o);
                break;
            }
            case 'R': case 0x81: {  // REDUCE, NEWOBJ
                PyRef a = pop(), fn = pop();
                st.push_back(call(fn, a));
                break;
            }
            case 0x92: {  // NEWOBJ_EX
                pop();
                PyRef a = pop(), fn = pop();
                st.push_back(call(fn, a));
                break;
            }
            case 'o': {
                auto v = pop_mark();
                if (v.empty()) throw std::runtime_error("pickle OBJ error");
                st.push_back(call(v[0], seq(K::Tuple, vector<PyRef>(v.begin() + 1, v.end()))));
                break;
            }
            case 'i': {
                auto g = mk(K::Global);
                string m = line();
                g->s = m + "." + line();
                st.push_back(call(g, seq(K::Tuple, pop_mark())));
                break;
            }
            case 'b': {
                PyRef s = pop();
                top()->state = s;
                break;
            }
            case 'p': memo[std::stoll(line())] = top(); break;
            case 'q': memo[rdv<uint8_t>()] = top(); break;
            case 'r': memo[rdv<uint32_t>()] = top(); break;
            case 0x94: memo[(int64_t)memo.size()] = top(); break;
            case 'g': case 'h': case 'j': {
                int64_t k = op == 'g' ? std::stoll(line()) : op == 'h' ? rdv<uint8_t>() : rdv<uint32_t>();
                auto it = memo.find(k);
                if (it == memo.end()) throw std::runtime_error("pickle memo error");
                st.push_back(it->second);
                break;
            }
            case 0x98: break;  // READONLY_BUFFER
            default: throw std::runtime_error(fmt("unsupported pickle opcode 0x%02x", op));
            }
        }
    }
};

static bool ends_with(const string &s, const string &t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// A numpy array from its pickled form: _frombuffer (protocol 5) or _reconstruct.
struct NpArr {
    vector<int64_t> shape;
    char kind = 0;  // 'i', 'u'
    int size = 0;
    bool big = false;
    PyRef buf;  // Bytes
    size_t count() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int64_t at(size_t k) const {
        const uint8_t *b = (const uint8_t *)buf->data + k * size;
        uint64_t v = 0;
        for (int j = 0; j < size; j++) v |= (uint64_t)b[big ? size - 1 - j : j] << (8 * j);
        if (kind == 'i' && size < 8 && (v >> (8 * size - 1) & 1)) v |= ~0ull << (8 * size);
        return (int64_t)v;
    }
};

static int64_t py_int(const PyRef &o) {
    if (!o || (o->kind != PyObj::Int && o->kind != PyObj::Bool)) throw std::runtime_error("pickle: expected int");
    return o->i;
}

static PyRef py_bytes(const PyRef &o) {
    if (o && o->kind == PyObj::Bytes) return o;
    // protocol < 3 from Python 3: _codecs.encode(latin1-decoded str, 'latin1')
    if (o && o->kind == PyObj::Call && o->func && ends_with(o->func->s, "encode") && o->args &&
        !o->args->items.empty() && o->args->items[0]->kind == PyObj::Str) {
        const string &u = o->args->items[0]->s;
        auto b = std::make_shared<PyObj>();
        b->kind = PyObj::Bytes;
        b->data = (char *)malloc(std::max<size_t>(u.size(), 1));
        for (size_t k = 0; k < u.size(); k++) {  // utf-8 code points < 256 -> latin1 bytes
            unsigned char c = u[k];
            b->data[b->n++] = c < 0x80 ? c : (char)(((c & 0x1f) << 6) | (u[++k] & 0x3f));
        }
        return b;
    }
    throw std::runtime_error("pickle: expected array data bytes");
}

static NpArr np_array(const PyRef &o) {
    if (!o || o->kind != PyObj::Call || !o->func) throw std::runtime_error("pickle: expected numpy array");
    PyRef dtype, shape, data;
    bool fortran = false;
    if (ends_with(o->func->s, "_frombuffer")) {
        auto &a = o->args->items;
        if (a.size() < 3) throw std::runtime_error("pickle: bad _frombuffer");
        data = a[0], dtype = a[1], shape = a[2];
        fortran = a.size() > 3 && a[3]->kind == PyObj::Str && a[3]->s == "F";
    } else if (ends_with(o->func->s, "_reconstruct")) {
        if (!o->state || o->state->items.size() < 4) throw std::runtime_error("pickle: bad ndarray state");
        auto &s = o->state->items;
        size_t k = s.size() == 5 ? 1 : 0;
        shape = s[k], dtype = s[k + 1], fortran = s[k + 2]->i != 0, data = s[k + 3];
    } else {
        throw std::runtime_error("pickle: unsupported array type " + o->func->s);
    }
    NpArr r;
    for (auto &d : shape->items) r.shape.push_back(py_int(d));
    if (!dtype || dtype->kind != PyObj::Call || dtype->args->items.empty())
        throw std::runtime_error("pickle: bad dtype");
    const string code = dtype->args->items[0]->s;  // e.g. "u4"
    if (code.size() < 2 || (code[0] != 'u' && code[0] != 'i')) throw std::runtime_error("pickle: unsupported dtype " + code);
    r.kind = code[0], r.size = std::stoi(code.substr(1));
    if (r.size != 1 && r.size != 2 && r.size != 4 && r.size != 8) throw std::runtime_error("pickle: bad dtype " + code);
    r.big = dtype->state && dtype->state->items.size() > 1 && dtype->state->items[1]->s == ">";
    if (fortran && r.shape.size() > 1) throw std::runtime_error("pickle: Fortran-order arrays not supported");
    r.buf = py_bytes(data);
    if (r.buf->n != r.count() * r.size) throw std::runtime_error("pickle: array size mismatch");
    return r;
}

// Pickle writer helpers
struct PickleOut {
    string b;
    void op(uint8_t c) { b += (char)c; }
    void raw(const void *p, size_t n) { b.append((const char *)p, n); }
    void integer(int64_t v) {
        if (v >= 0 && v < 256) op('K'), op((uint8_t)v);
        else if (v >= INT32_MIN && v <= INT32_MAX) { int32_t x = (int32_t)v; op('J'), raw(&x, 4); }
        else op(0x8a), op(8), raw(&v, 8);
    }
    void str(const string &s) {
        if (s.size() < 256) op(0x8c), op((uint8_t)s.size());
        else { uint32_t n = s.size(); op('X'), raw(&n, 4); }
        raw(s.data(), s.size());
    }
    void global(const char *mod, const char *name) {
        op('c');
        b += mod, b += '\n', b += name, b += '\n';
    }
    // numpy array in the classic _reconstruct form; data follows (bytes_len) then np_end()
    void np_begin(const vector<int64_t> &shape, const char *code, size_t nbytes) {
        global("numpy.core.multiarray", "_reconstruct");
        global("numpy", "ndarray");
        integer(0), op(0x85);
        op('C'), op(1), op('b');
        op(0x87), op('R');
        op('('), integer(1);
        op('(');
        for (auto d : shape) integer(d);
        op('t');
        global("numpy", "dtype");
        str(code), op(0x89), op(0x88), op(0x87), op('R');
        op('('), integer(3), str("<"), op('N'), op('N'), op('N'), integer(-1), integer(-1), integer(0), op('t'), op('b');
        op(0x89);
        if (nbytes < (1ull << 32)) { uint32_t n = nbytes; op('B'), raw(&n, 4); }
        else { uint64_t n = nbytes; op(0x8e), raw(&n, 8); }
    }
    void np_end() { op('t'), op('b'); }
};

// Multi-member gzip, members compressed in parallel (gzip, zlib and Python's
// gzip module all read concatenated members as one stream).
static void write_gzip(const string &fn, const vector<std::pair<const char *, size_t>> &parts) {
    const size_t BLK = 16u << 20;
    vector<std::pair<const char *, size_t>> blocks;
    for (auto [p, n] : parts)
        for (size_t off = 0; off < n; off += BLK) blocks.push_back({p + off, std::min(BLK, n - off)});
    vector<string> z(blocks.size());
    std::atomic<bool> err{false};
    parallel_for((int)blocks.size(), g_threads, [&](int k) {
        z_stream s{};
        if (deflateInit2(&s, 6, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) { err = true; return; }
        s.next_in = (Bytef *)blocks[k].first, s.avail_in = (uInt)blocks[k].second;
        char buf[1 << 16];  // mostly-empty tables compress ~1000x: grow output as needed
        int r;
        do {
            s.next_out = (Bytef *)buf, s.avail_out = sizeof buf;
            r = deflate(&s, Z_FINISH);
            z[k].append(buf, sizeof buf - s.avail_out);
        } while (r == Z_OK);
        if (r != Z_STREAM_END) err = true;
        deflateEnd(&s);
    });
    if (err) throw std::runtime_error("gzip compression failed for " + fn);
    FILE *f = fopen(fn.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + fn);
    for (auto &c : z) fwrite(c.data(), 1, c.size(), f);
    if (fclose(f) != 0) throw std::runtime_error("error writing " + fn);
}

// ---------------------------------------------------------------- hash table

struct HashTable {
    int hashbits = 20, depth = 100, maxtimebits = 14;
    uint32_t *table = nullptr;  // calloc: untouched buckets cost no RAM
    vector<int32_t> counts;
    vector<string> names;  // "" = removed slot (None in Python)
    vector<uint32_t> hashesperid;
    std::unordered_map<string, int> ids;
    int samplerate = 0;
    bool dirty = true;
    std::mt19937_64 rng{1234};

    HashTable() = default;
    HashTable(const HashTable &) = delete;
    ~HashTable() { free(table); }

    void alloc(int hb, int dp, int mtb) {
        hashbits = hb, depth = dp, maxtimebits = mtb;
        free(table);
        table = (uint32_t *)calloc((size_t)depth << hashbits, sizeof(uint32_t));
        if (!table) throw std::runtime_error("cannot allocate hash table");
        counts.assign((size_t)1 << hashbits, 0);
    }
    uint32_t *bucket(uint32_t h) const { return table + (size_t)h * depth; }

    int name_to_id(const string &name, bool add) {
        auto it = ids.find(name);
        if (it != ids.end()) return it->second;
        if (!add) throw std::runtime_error("name " + name + " not found");
        // reuse the first removed slot, like names.index(None)
        int id = ids.size() == names.size() ? (int)names.size()
                                            : (int)(std::find(names.begin(), names.end(), string()) - names.begin());
        if (id == (int)names.size()) names.push_back(name), hashesperid.push_back(0);
        else names[id] = name, hashesperid[id] = 0;
        ids[name] = id;
        return id;
    }

    void store(const string &name, const vector<TH> &hs) {
        int id = name_to_id(name, true);
        const uint32_t hmask = (1u << hashbits) - 1, tmask = (1u << maxtimebits) - 1;
        const uint32_t idval = (uint32_t)(id + 1) << maxtimebits;
        for (auto [t, h] : hs) {
            uint32_t hh = (uint32_t)h & hmask;
            int32_t cnt = counts[hh];
            uint32_t val = idval + ((uint32_t)t & tmask);
            if (cnt < depth) bucket(hh)[cnt] = val;
            else if (uint64_t slot = rng() % ((uint64_t)cnt + 1); slot < (uint64_t)depth) bucket(hh)[slot] = val;
            counts[hh] = cnt + 1;
        }
        hashesperid[id] += hs.size();
        dirty = true;
    }

    int64_t total() const {
        int64_t s = 0;
        for (auto c : counts) s += c;
        return s;
    }
    int64_t dropped() const {
        int64_t s = 0;
        for (auto c : counts) s += std::max(0, c - depth);
        return s;
    }
    int nfiles() const {
        return (int)std::count_if(names.begin(), names.end(), [](auto &s) { return !s.empty(); });
    }
    string summary(const char *verb, const string &fn) const {
        int64_t n = total();
        return fmt("%s fprints for %d files ( %lld hashes) %s %s (%.2f%% dropped)", verb, nfiles(),
                   (long long)n, verb[0] == 'S' ? "to" : "from", fn.c_str(),
                   100.0 * dropped() / std::max<int64_t>(1, n));
    }

    // .pklz (gzipped Python pickle, same as Python audfprint) when the name
    // ends in .pklz, else the compact native format.
    void save(const string &fn) {
        if (ends_with(fn, ".pklz")) save_pklz(fn);
        else save_native(fn);
        dirty = false;
        printf("%s\n", summary("Saved", fn).c_str());
    }

    void load(const string &fn) {
        unsigned char magic[2] = {0, 0};
        if (FILE *f = fopen(fn.c_str(), "rb")) {
            if (fread(magic, 1, 2, f) != 2) magic[0] = 0;
            fclose(f);
        } else {
            throw std::runtime_error("cannot read " + fn);
        }
        if (ends_with(fn, ".mat")) throw std::runtime_error("Matlab .mat databases are not supported: " + fn);
        if (magic[0] == 0x1f && magic[1] == 0x8b) load_pklz(fn);
        else load_native(fn);
        dirty = false;
        printf("%s\n", summary("Read", fn).c_str());
    }

    void save_pklz(const string &fn) const {
        const size_t nb = (size_t)1 << hashbits, tbytes = nb * depth * sizeof(uint32_t);
        PickleOut a, c;  // pickle before / after the raw table bytes
        a.op(0x80), a.op(4);
        a.global("hash_table", "HashTable");
        a.op(')'), a.op(0x81);
        a.op('}'), a.op('(');
        a.str("hashbits"), a.integer(hashbits);
        a.str("depth"), a.integer(depth);
        a.str("maxtimebits"), a.integer(maxtimebits);
        a.str("table"), a.np_begin({(int64_t)nb, depth}, "u4", tbytes);
        c.np_end();
        c.str("counts"), c.np_begin({(int64_t)nb}, "i4", nb * 4), c.raw(counts.data(), nb * 4), c.np_end();
        c.str("names"), c.op(']');
        if (!names.empty()) {
            c.op('(');
            for (auto &n : names) n.empty() ? c.op('N') : c.str(n);
            c.op('e');
        }
        c.str("hashesperid"), c.np_begin({(int64_t)hashesperid.size()}, "u4", hashesperid.size() * 4);
        c.raw(hashesperid.data(), hashesperid.size() * 4), c.np_end();
        c.str("params"), c.op('}');
        if (samplerate) c.op('('), c.str("samplerate"), c.integer(samplerate), c.op('u');
        c.str("ht_version"), c.integer(20170724);
        c.str("dirty"), c.op(0x89);
        c.op('u'), c.op('b'), c.op('.');
        write_gzip(fn, {{a.b.data(), a.b.size()}, {(const char *)table, tbytes}, {c.b.data(), c.b.size()}});
    }

    void load_pklz(const string &fn) {
        gzFile f = gzopen(fn.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot read " + fn);
        gzbuffer(f, 1 << 20);
        PyRef top;
        try {
            top = Unpickler(f).load();
        } catch (...) {
            gzclose(f);
            throw;
        }
        gzclose(f);
        // NEWOBJ(HashTable) or copyreg._reconstructor(HashTable, object, None), then BUILD dict
        if (!top || top->kind != PyObj::Call || !top->func || !top->state || top->state->kind != PyObj::Dict)
            throw std::runtime_error(fn + " is not an audfprint hash table pickle");
        string cls = top->func->s;
        if (ends_with(cls, "_reconstructor") && top->args && !top->args->items.empty()) cls = top->args->items[0]->s;
        if (!ends_with(cls, "HashTable")) throw std::runtime_error(fn + " holds a " + cls + ", not a HashTable");
        const PyObj &d = *top->state;
        auto need = [&](const char *k) {
            PyRef v = d.get(k);
            if (!v) throw std::runtime_error(fn + ": pickle has no '" + k + "'");
            return v;
        };
        const int64_t ver = py_int(need("ht_version"));
        if (ver < 20140920)
            throw std::runtime_error("Version of " + fn + " is " + std::to_string(ver) + " which is not at least 20140920");
        const int64_t hb = py_int(need("hashbits")), dp = py_int(need("depth"));
        int64_t mtb;
        if (PyRef m = d.get("maxtimebits")) {
            mtb = py_int(m);
        } else {
            int64_t mt = py_int(need("maxtime"));
            mtb = std::lround(std::log2((double)std::max<int64_t>(1, mt)));
            if ((1ll << mtb) != mt) throw std::runtime_error("maxval must be a power of 2, not " + std::to_string(mt));
        }
        if (hb < 1 || hb > 30 || dp < 1 || dp > (1 << 20) || mtb < 1 || mtb > 31)
            throw std::runtime_error(fn + ": bad hash table parameters");
        NpArr t = np_array(need("table")), c = np_array(need("counts"));
        const size_t nb = (size_t)1 << hb;
        if (t.shape.size() != 2 || (size_t)t.shape[0] != nb || t.shape[1] != dp || c.count() != nb)
            throw std::runtime_error(fn + ": table shape does not match parameters");
        hashbits = (int)hb, depth = (int)dp, maxtimebits = (int)mtb;
        free(table);
        if (t.kind == 'u' && t.size == 4 && !t.big) {  // adopt the decoded buffer, no copy
            table = (uint32_t *)t.buf->data;
            t.buf->data = nullptr;
        } else {
            table = (uint32_t *)calloc(nb * dp, sizeof(uint32_t));
            if (!table) throw std::bad_alloc();
            for (size_t k = 0; k < nb * dp; k++) table[k] = (uint32_t)t.at(k);
        }
        counts.resize(nb);
        for (size_t k = 0; k < nb; k++) counts[k] = (int32_t)c.at(k);
        names.clear();
        ids.clear();
        for (auto &x : need("names")->items)  // None / [] (Matlab-derived) = empty slot
            names.push_back(x->kind == PyObj::Str ? x->s : x->kind == PyObj::Bytes ? string(x->data, x->n) : string());
        hashesperid.assign(names.size(), 0);
        PyRef hp = need("hashesperid");
        if (hp->kind == PyObj::List || hp->kind == PyObj::Tuple) {
            for (size_t k = 0; k < std::min(hp->items.size(), names.size()); k++) hashesperid[k] = (uint32_t)py_int(hp->items[k]);
        } else {
            NpArr h = np_array(hp);
            for (size_t k = 0; k < std::min(h.count(), names.size()); k++) hashesperid[k] = (uint32_t)h.at(k);
        }
        for (size_t i = 0; i < names.size(); i++)
            if (!names[i].empty()) ids[names[i]] = (int)i;
        samplerate = 0;
        if (PyRef p = d.get("params"))
            if (PyRef s = p->get("samplerate")) samplerate = (int)py_int(s);
        if (ver < 20170724) {  // old format: ids in table started at 0
            printf("Loading database version %lld in compatibility mode.\n", (long long)ver);
            const uint32_t off = 1u << maxtimebits;
            for (size_t k = 0; k < nb * dp; k++)
                if (table[k]) table[k] += off;
        }
    }

    // Native format: magic, params, names, then each non-empty bucket as
    // (hash, count, min(count, depth) entries).
    void save_native(const string &fn) const {
        FILE *f = fopen(fn.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + fn);
        int32_t hdr[5] = {hashbits, depth, maxtimebits, samplerate, (int32_t)names.size()};
        fwrite("AFPCPP02", 1, 8, f);
        fwrite(hdr, 4, 5, f);
        for (size_t i = 0; i < names.size(); i++) {
            uint32_t len = names[i].size();
            fwrite(&len, 4, 1, f);
            fwrite(names[i].data(), 1, len, f);
            fwrite(&hashesperid[i], 4, 1, f);
        }
        for (uint32_t h = 0; h < counts.size(); h++)
            if (counts[h]) {
                int32_t hc[2] = {(int32_t)h, counts[h]};
                fwrite(hc, 4, 2, f);
                fwrite(bucket(h), 4, std::min(counts[h], depth), f);
            }
        if (fclose(f) != 0) throw std::runtime_error("error writing " + fn);
    }

    void load_native(const string &fn) {
        FILE *f = fopen(fn.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot read " + fn);
        char magic[8];
        int32_t hdr[5];
        if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "AFPCPP02", 8) || fread(hdr, 4, 5, f) != 5 ||
            hdr[0] < 1 || hdr[0] > 30 || hdr[1] < 1 || hdr[2] < 1 || hdr[2] > 31 || hdr[4] < 0)
            throw std::runtime_error(fn + " is not an audfprint-cpp database");
        alloc(hdr[0], hdr[1], hdr[2]);
        samplerate = hdr[3];
        names.resize(hdr[4]);
        hashesperid.resize(hdr[4]);
        bool ok = true;
        for (int i = 0; i < hdr[4] && ok; i++) {
            uint32_t len;
            ok = fread(&len, 4, 1, f) == 1 && len < (1u << 20);
            if (!ok) break;
            names[i].resize(len);
            ok = fread(names[i].data(), 1, len, f) == len && fread(&hashesperid[i], 4, 1, f) == 1;
            if (!names[i].empty()) ids[names[i]] = i;
        }
        for (int32_t hc[2]; ok && fread(hc, 4, 2, f) == 2;) {
            ok = (uint32_t)hc[0] < counts.size() && hc[1] > 0;
            if (!ok) break;
            counts[hc[0]] = hc[1];
            size_t k = std::min(hc[1], depth);
            ok = fread(bucket(hc[0]), 4, k, f) == k;
        }
        fclose(f);
        if (!ok) throw std::runtime_error("corrupt database " + fn);
    }

    void merge(HashTable &o) {
        if (o.maxtimebits != maxtimebits) throw std::runtime_error("merge: maxtimebits differ");
        const uint32_t idoffset = (uint32_t)names.size() << maxtimebits;
        for (size_t i = 0; i < o.names.size(); i++) {
            names.push_back(o.names[i]);
            hashesperid.push_back(o.hashesperid[i]);
            if (!o.names[i].empty()) ids[o.names[i]] = names.size() - 1;
        }
        vector<uint32_t> all;
        const size_t nb = std::min(counts.size(), o.counts.size());
        for (size_t h = 0; h < nb; h++) {
            if (!o.counts[h]) continue;
            all.assign(bucket(h), bucket(h) + std::min(counts[h], depth));
            for (int j = 0; j < std::min(o.counts[h], o.depth); j++) all.push_back(o.bucket(h)[j] + idoffset);
            if ((int)all.size() > depth) {
                std::shuffle(all.begin(), all.end(), rng);
                std::copy(all.begin(), all.begin() + depth, bucket(h));
                counts[h] += o.counts[h];
            } else {
                std::copy(all.begin(), all.end(), bucket(h));
                counts[h] = all.size();
            }
        }
        dirty = true;
    }

    void remove(const string &name) {
        int id = name_to_id(name, false);
        const uint32_t idv = id + 1;
        int64_t removed = 0;
        for (size_t h = 0; h < counts.size(); h++) {
            uint32_t *b = bucket(h);
            int n = std::min(counts[h], depth), k = 0;
            if (std::none_of(b, b + n, [&](uint32_t v) { return (v >> maxtimebits) == idv; })) continue;
            for (int j = 0; j < n; j++)
                if ((b[j] >> maxtimebits) != idv) b[k++] = b[j];
            removed += n - k;
            std::fill(b + k, b + depth, 0);
            counts[h] = k;
        }
        names[id].clear();
        ids.erase(name);
        hashesperid[id] = 0;
        dirty = true;
        printf("Removed %s ( %lld hashes).\n", name.c_str(), (long long)removed);
    }
};

// ---------------------------------------------------------------- matching

struct Hit { int32_t id, dt, hash, otime; };
struct Result { int32_t id, count, mode, raw, rank, tmin, tmax; };

struct Matcher {
    int window = 2, threshcount = 5, max_returns = 1, search_depth = 100, max_alignments_per_id = 100;
    bool sort_by_time = false, exact_count = false, find_time_range = false;
    double time_quantile = 0.05;

    static vector<Hit> get_hits(const HashTable &ht, const vector<TH> &q) {
        vector<Hit> hits;
        const uint32_t hmask = (1u << ht.hashbits) - 1, tmask = (1u << ht.maxtimebits) - 1;
        for (auto [t, h] : q) {
            uint32_t hh = (uint32_t)h & hmask;
            int n = std::min(ht.counts[hh], ht.depth);
            const uint32_t *b = ht.bucket(hh);
            for (int j = 0; j < n; j++)
                hits.push_back({(int32_t)(b[j] >> ht.maxtimebits) - 1, (int32_t)(b[j] & tmask) - t, (int32_t)hh, t});
        }
        return hits;
    }

    // [match_times[int(n*q)], match_times[int(n*(1-q)) - 1]] with Python negative indexing
    std::pair<int, int> time_range(const vector<int32_t> &mt) const {
        long n = mt.size();
        long lo = (long)(n * time_quantile), hi = (long)(n * (1.0 - time_quantile)) - 1;
        if (hi < 0) hi += n;
        return {mt[lo], mt[hi]};
    }

    vector<Result> match_hashes(const HashTable &ht, const vector<TH> &q) const {
        vector<Hit> hits = get_hits(ht, q);
        vector<Result> res;
        if (hits.empty()) return res;
        // _best_count_ids
        const int nids = ht.names.size();
        vector<int32_t> raw(nids, 0);
        for (auto &h : hits) raw[h.id]++;
        vector<int> ids;
        for (int i = 0; i < nids; i++)
            if (raw[i]) ids.push_back(i);
        // argsort(wtd)[::-1]; ties resolved like a stable ascending sort, reversed
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
            return (double)raw[a] / ht.hashesperid[a] < (double)raw[b] / ht.hashesperid[b];
        });
        std::reverse(ids.begin(), ids.end());
        int above = std::count_if(ids.begin(), ids.end(), [&](int i) { return raw[i] > threshcount; });
        ids.resize(std::min(above, search_depth));
        if (ids.empty()) return res;

        // group hits by id, each group in original-time order
        std::stable_sort(hits.begin(), hits.end(), [](auto &a, auto &b) { return a.otime < b.otime; });
        vector<int> slot(nids, -1);
        for (size_t r = 0; r < ids.size(); r++) slot[ids[r]] = r;
        vector<vector<Hit>> by(ids.size());
        int32_t mintime = INT32_MAX, maxotime = 0;
        for (auto &h : hits) {
            mintime = std::min(mintime, h.dt);
            maxotime = std::max(maxotime, h.otime);
            if (slot[h.id] >= 0) by[slot[h.id]].push_back(h);
        }
        auto times_in = [&](const vector<Hit> &g, int mode) {
            vector<int32_t> mt;
            for (auto &h : g)
                if (h.dt >= mode - window && h.dt <= mode + window) mt.push_back(h.otime);
            return mt;
        };
        auto locmax = [](const vector<int32_t> &v, size_t i) {
            return (i == 0 || v[i] >= v[i - 1]) && !(i + 1 < v.size() && v[i + 1] >= v[i]);
        };
        // timebits = max(1, ceil(log2(max otime))) as in _unique_match_hashes
        const int timebits = std::max(1, (int)std::ceil(std::log(std::max(1, maxotime)) / std::log(2.0)));

        for (size_t r = 0; r < ids.size(); r++) {
            const int id = ids[r];
            const vector<Hit> &g = by[r];
            if (!exact_count) {  // _approx_match_counts
                int32_t mx = 0;
                for (auto &h : g) mx = std::max(mx, h.dt - mintime);
                vector<int32_t> bc(mx + 1, 0);
                for (auto &h : g) bc[h.dt - mintime]++;
                vector<int32_t> filt(bc.size(), 0);
                for (size_t i = 0; i < bc.size(); i++)
                    if (locmax(bc, i)) filt[i] = bc[i];
                for (int found = 0;;) {
                    long mode = std::max_element(filt.begin(), filt.end()) - filt.begin();
                    if (filt[mode] <= threshcount) break;
                    long lo = std::max(0L, mode - window), hi = std::min<long>(mode + window + 1, bc.size());
                    int count = 0;
                    for (long i = lo; i < hi; i++) count += bc[i];
                    Result rr{id, count, (int32_t)(mode + mintime), raw[id], (int32_t)r, 0, 0};
                    if (find_time_range) std::tie(rr.tmin, rr.tmax) = time_range(times_in(g, rr.mode));
                    res.push_back(rr);
                    std::fill(filt.begin() + lo, filt.begin() + hi, 0);
                    if (++found > max_alignments_per_id) break;
                }
            } else {  // _exact_match_counts
                int32_t dmin = INT32_MAX, dmax = INT32_MIN;
                for (auto &h : g) dmin = std::min(dmin, h.dt), dmax = std::max(dmax, h.dt);
                vector<int32_t> fv(dmax - dmin + 1, 0);
                for (auto &h : g) fv[h.dt - dmin]++;
                vector<int64_t> keys;
                for (size_t i = 0; i < fv.size(); i++) {
                    if (!locmax(fv, i) || fv[i] < threshcount) continue;
                    int mode = dmin + (int)i;
                    keys.clear();
                    for (auto &h : g)
                        if (std::abs(h.dt - mode) <= window) keys.push_back(h.otime + ((int64_t)h.hash << timebits));
                    std::sort(keys.begin(), keys.end());
                    int filtcount = std::unique(keys.begin(), keys.end()) - keys.begin();
                    if (filtcount < threshcount) continue;
                    Result rr{id, filtcount, mode, raw[id], (int32_t)r, 0, 0};
                    if (find_time_range) std::tie(rr.tmin, rr.tmax) = time_range(times_in(g, mode));
                    res.push_back(rr);
                }
            }
        }
        std::stable_sort(res.begin(), res.end(), [](auto &a, auto &b) { return a.count > b.count; });
        return res;
    }
};

// ---------------------------------------------------------------- CLI

static const char *USAGE = R"(
Landmark-based audio fingerprinting (C++ port of audfprint).
Create a new fingerprint dbase with "new",
append new files to an existing database with "add",
or identify noisy query excerpts with "match".
"precompute" writes a *.afpt file under precompdir
with precomputed fingerprint for each input wav file.
"merge" combines previously-created databases into
an existing database; "newmerge" combines existing
databases to create a new one.

Usage: audfprint (new | add | match | precompute | merge | newmerge | list | remove) [options] [<file>]...

Options:
  -d <dbase>, --dbase <dbase>     Fingerprint database file
  -n <dens>, --density <dens>     Target hashes per second [default: 20.0]
  -h <bits>, --hashbits <bits>    How many bits in each hash [default: 20]
  -b <val>, --bucketsize <val>    Number of entries per bucket [default: 100]
  -t <val>, --maxtime <val>       Largest time value stored [default: 16384]
  -u <val>, --maxtimebits <val>   maxtime as a number of bits (16384 == 14 bits)
  -r <val>, --samplerate <val>    Resample input files to this [default: 11025]
  -p <dir>, --precompdir <dir>    Save precomputed files under this dir [default: .]
  -i <val>, --shifts <val>        Use this many subframe shifts building fp [default: 0]
  -w <val>, --match-win <val>     Maximum tolerable frame skew to count as a match [default: 2]
  -N <val>, --min-count <val>     Minimum number of matching landmarks to count as a match [default: 5]
  -x <val>, --max-matches <val>   Maximum number of matches to report for each query [default: 1]
  -X, --exact-count               Flag to use more precise (but slower) match counting
  -R, --find-time-range           Report the time support of each match
  -Q, --time-quantile <val>       Quantile at extremes of time support [default: 0.05]
  -S <val>, --freq-sd <val>       Frequency peak spreading SD in bins [default: 30.0]
  -F <val>, --fanout <val>        Max number of hash pairs per peak [default: 3]
  -P <val>, --pks-per-frame <val>  Maximum number of peaks per frame [default: 5]
  -D <val>, --search-depth <val>  How far down to search raw matching track list [default: 100]
  -H <val>, --ncores <val>        Number of threads to use, 0 = all [default: 0]
  -o <name>, --opfile <name>      Write output (matches) to this file, not stdout [default: ]
  -K, --precompute-peaks          Precompute just landmarks (else full hashes)
  -k, --skip-existing             On precompute, skip items if output file already exists
  -C, --continue-on-error         Keep processing despite errors reading input
  -l, --list                      Input files are lists, not audio
  -T, --sortbytime                Sort multiple hits per file by time (instead of score)
  -v <val>, --verbose <val>       Verbosity level [default: 1]
  -W <dir>, --wavdir <dir>        Find sound files under this dir [default: ]
  -V <ext>, --wavext <ext>        Extension to add to wav file names [default: ]
  --version                       Report version number
  --help                          Print this message
)";

struct Args {
    string cmd;
    std::map<string, string> opt;  // keyed by long name
    vector<string> files;
    bool flag(const string &k) const { return opt.count(k) && opt.at(k) == "1"; }
    string str(const string &k) const { return opt.count(k) ? opt.at(k) : ""; }
    int i(const string &k) const { return std::stoi(str(k)); }
    double f(const string &k) const { return std::stod(str(k)); }
};

static Args parse_args(int argc, char **argv) {
    struct O { string s, l; bool takes; };
    vector<O> opts;
    Args a;
    // Option table + defaults straight from the usage text, docopt style.
    for (const char *p = strstr(USAGE, "Options:"); (p = strchr(p, '\n')); p++) {
        const char *e = strchr(p + 1, '\n');
        string line(p + 1, e ? e : p + 1 + strlen(p + 1));
        size_t q = line.find_first_not_of(' ');
        if (q == string::npos || line[q] != '-') continue;
        string spec = line.substr(q, line.find("  ", q) - q);
        O o{"", "", spec.find('<') != string::npos};
        for (size_t k = 0; k < spec.size();) {
            size_t end = spec.find_first_of(", =", k);
            string tok = spec.substr(k, end - k);
            if (tok.rfind("--", 0) == 0) o.l = tok;
            else if (tok.size() == 2 && tok[0] == '-') o.s = tok;
            k = end == string::npos ? spec.size() : end + 1;
        }
        size_t dpos = line.find("[default: ");
        if (dpos != string::npos) a.opt[o.l] = line.substr(dpos + 10, line.find(']', dpos) - dpos - 10);
        opts.push_back(o);
    }
    const vector<string> cmds = {"new", "add", "match", "precompute", "merge", "newmerge", "list", "remove"};
    for (int k = 1; k < argc; k++) {
        string t = argv[k], val;
        if (t == "--help") { puts(USAGE); exit(0); }
        if (t == "--version") { puts("20150406-cpp"); exit(0); }
        if (t.size() > 1 && t[0] == '-') {
            size_t eq = t.find('=');
            if (eq != string::npos) val = t.substr(eq + 1), t = t.substr(0, eq);
            auto it = std::find_if(opts.begin(), opts.end(), [&](auto &o) { return o.s == t || o.l == t; });
            if (it == opts.end()) throw std::runtime_error("unknown option " + t);
            if (it->takes && eq == string::npos) {
                if (++k >= argc) throw std::runtime_error("option " + t + " needs a value");
                val = argv[k];
            }
            a.opt[it->l] = it->takes ? val : "1";
        } else if (a.cmd.empty() && std::find(cmds.begin(), cmds.end(), t) != cmds.end()) {
            a.cmd = t;
        } else {
            a.files.push_back(t);
        }
    }
    if (a.cmd.empty()) throw std::runtime_error("must specify exactly one command\n" + string(USAGE));
    return a;
}

static vector<string> expand_files(const Args &a) {
    vector<string> out;
    auto add = [&](const string &f) { out.push_back(a.str("--wavdir").empty() ? f + a.str("--wavext") : (fs::path(a.str("--wavdir")) / (f + a.str("--wavext"))).string()); };
    for (auto &f : a.files) {
        if (!a.flag("--list")) { add(f); continue; }
        std::ifstream in(f);
        for (string line; std::getline(in, line);) add(line);
    }
    return out;
}

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        const string cmd = args.cmd;
        const double t0 = wall();
        int ncores = args.i("--ncores");
        if (ncores <= 0) ncores = std::max(1u, std::thread::hardware_concurrency());
        g_threads = ncores;
        av_log_set_level(AV_LOG_ERROR);

        FILE *opf = stdout;
        if (!args.str("--opfile").empty() && !(opf = fopen(args.str("--opfile").c_str(), "w")))
            throw std::runtime_error("cannot write " + args.str("--opfile"));
        auto report = [&](const vector<string> &msgs) {
            for (auto &m : msgs) fprintf(opf, "%s\n", m.c_str());
            fflush(opf);
        };

        int maxtimebits = args.str("--maxtimebits").empty() ? (int)std::lround(std::log2(args.f("--maxtime"))) : args.i("--maxtimebits");
        if (args.str("--maxtimebits").empty() && (1 << maxtimebits) != args.i("--maxtime"))
            throw std::runtime_error("maxval must be a power of 2, not " + args.str("--maxtime"));

        Analyzer an;
        an.density = args.f("--density");
        an.maxpksperframe = args.i("--pks-per-frame");
        an.maxpairsperpeak = args.i("--fanout");
        an.f_sd = args.f("--freq-sd");
        an.shifts = args.i("--shifts");
        an.target_sr = args.i("--samplerate");
        if (an.shifts == 0) an.shifts = cmd == "match" ? 4 : 1;
        an.fail_on_error = !args.flag("--continue-on-error");
        an.init();

        Matcher mt;
        mt.window = args.i("--match-win");
        mt.threshcount = args.i("--min-count");
        mt.max_returns = args.i("--max-matches");
        mt.search_depth = args.i("--search-depth");
        mt.sort_by_time = args.flag("--sortbytime");
        mt.exact_count = args.flag("--exact-count");
        mt.find_time_range = args.flag("--find-time-range");
        mt.time_quantile = args.f("--time-quantile");

        HashTable ht;
        const string dbname = args.str("--dbase");
        if (cmd != "precompute") {
            if (dbname.empty()) throw std::runtime_error("dbase name must be provided if not precompute");
            if (cmd == "new" || cmd == "newmerge") {
                if (fs::path(dbname).has_parent_path()) fs::create_directories(fs::path(dbname).parent_path());
                ht.alloc(args.i("--hashbits"), args.i("--bucketsize"), maxtimebits);
                if (cmd == "new") ht.samplerate = an.target_sr;
            } else {
                printf("%s Reading hash table %s\n", now_ctime().c_str(), dbname.c_str());
                ht.load(dbname);
                if (cmd != "merge" && ht.samplerate && ht.samplerate != an.target_sr)
                    printf("db samplerate overridden to  %d\n", an.target_sr);
            }
        }

        const vector<string> files = expand_files(args);
        double totdur = 0.0;
        int nprocessed = 0;
        const double t_hop = (double)N_HOP / an.target_sr;

        // Analyze files in parallel batches; consume results in input order.
        struct Out { vector<TH> h; vector<Peak> pk; double dur = 0; string err; vector<string> msgs; };
        auto run_batched = [&](const std::function<void(int, Out &, int)> &work,
                               const std::function<void(int, Out &)> &consume) {
            const int B = std::max(ncores * 4, 64);
            for (int b0 = 0; b0 < (int)files.size(); b0 += B) {
                int nb = std::min<int>(B, files.size() - b0);
                vector<Out> outs(nb);
                int inner = std::max(1, ncores / nb);
                parallel_for(nb, ncores, [&](int k) {
                    try { work(b0 + k, outs[k], inner); } catch (std::exception &e) { outs[k].err = e.what(); }
                });
                for (int k = 0; k < nb; k++) {
                    Out &o = outs[k];
                    if (!o.err.empty()) {
                        if (an.fail_on_error) {
                            fprintf(stderr, "%s\n", o.err.c_str());
                            throw std::runtime_error("wavfile2peaks: Error reading " + files[b0 + k]);
                        }
                        printf("wavfile2peaks: Error reading %s skipping\n", files[b0 + k].c_str());
                        o = Out{};
                    }
                    totdur += o.dur;
                    nprocessed++;
                    consume(b0 + k, o);
                }
            }
        };

        if (cmd == "new" || cmd == "add") {
            int64_t tothashes = 0;
            run_batched([&](int i, Out &o, int inner) { o.h = an.file2hashes(files[i], an.shifts, inner, o.dur); },
                        [&](int i, Out &o) {
                            report({now_ctime() + " ingesting #" + std::to_string(i) + ": " + files[i] + " ..."});
                            ht.store(files[i], o.h);
                            tothashes += o.h.size();
                        });
            report({fmt("Added %lld hashes (%.1f hashes/sec)", (long long)tothashes, tothashes / totdur)});
        } else if (cmd == "match") {
            run_batched(
                [&](int i, Out &o, int inner) {
                    const string &qry = files[i];
                    vector<TH> q = an.file2hashes(qry, an.shifts, inner, o.dur);
                    double durd = q.empty() ? 0.0 : (double)N_HOP * q.back().t / an.target_sr;
                    o.msgs.push_back(fmt("%s Analyzed #%d %s of %.3f s to %zu hashes", now_ctime().c_str(), i, qry.c_str(), durd, q.size()));
                    vector<Result> r = mt.match_hashes(ht, q);
                    if (mt.sort_by_time) std::stable_sort(r.begin(), r.end(), [](auto &a, auto &b) { return a.mode > b.mode; });
                    if ((int)r.size() > mt.max_returns) r.resize(mt.max_returns);
                    string qrymsg = qry + fmt(" %.1f sec %zu raw hashes", durd, q.size());
                    if (r.empty()) o.msgs.push_back("NOMATCH " + qrymsg);
                    for (auto &x : r) {
                        string m = mt.find_time_range
                            ? fmt("Matched %6.1f s starting at %6.1f s in %s to time %6.1f s in %s", (x.tmax - x.tmin) * t_hop,
                                  x.tmin * t_hop, qry.c_str(), (x.tmin + x.mode) * t_hop, ht.names[x.id].c_str())
                            : fmt("Matched %s as %s at %6.1f s", qrymsg.c_str(), ht.names[x.id].c_str(), x.mode * t_hop);
                        o.msgs.push_back(m + fmt(" with %5d of %5d common hashes at rank %2d", x.count, x.raw, x.rank));
                    }
                },
                [&](int, Out &o) {
                    if (!o.msgs.empty()) printf("%s\n", o.msgs[0].c_str());
                    report(vector<string>(o.msgs.begin() + std::min<size_t>(1, o.msgs.size()), o.msgs.end()));
                });
        } else if (cmd == "precompute") {
            const bool peaks = args.flag("--precompute-peaks");
            const string strip = args.str("--wavdir");
            run_batched(
                [&](int i, Out &o, int inner) {
                    string fn = files[i], tail = fn;
                    if (!strip.empty() && fn.rfind(strip, 0) == 0) tail = fn.substr(strip.size());
                    string rel;  // drop '.', '..' and empty components
                    for (size_t k = 0; k <= tail.size();) {
                        size_t e = std::min(tail.find('/', k), tail.size());
                        string c = tail.substr(k, e - k);
                        if (!c.empty() && c != "." && c != "..") rel += (rel.empty() ? "" : "/") + c;
                        k = e + 1;
                    }
                    string root = rel.substr(0, rel.size() - fs::path(rel).extension().string().size());
                    string opf = (fs::path(args.str("--precompdir")) / (root + (peaks ? ".afpk" : ".afpt"))).string();
                    o.msgs.push_back(now_ctime() + " precomputing " + (peaks ? "peaks" : "hashes") + " for " + fn + " ...");
                    if (args.flag("--skip-existing") && fs::is_regular_file(opf)) {
                        o.msgs.push_back("file " + opf + " exists (and --skip-existing); skipping");
                        return;
                    }
                    size_t n;
                    if (peaks) o.pk = an.file2peaks(fn, inner, o.dur), n = o.pk.size();
                    else o.h = an.file2hashes(fn, an.shifts, inner, o.dur), n = o.h.size();
                    if (n == 0) {
                        o.msgs.push_back("Zero length analysis for " + fn + " -- not saving.");
                        return;
                    }
                    if (fs::path(opf).has_parent_path()) fs::create_directories(fs::path(opf).parent_path());
                    if (peaks) save_pairs(opf, PEAK_MAGIC, o.pk);
                    else save_pairs(opf, HASH_MAGIC, o.h);
                    o.msgs.push_back(fmt("wrote %s ( %zu %s, %.3f sec)", opf.c_str(), n, peaks ? "peaks" : "hashes", o.dur));
                },
                [&](int, Out &o) {
                    if (!o.msgs.empty()) printf("%s\n", o.msgs[0].c_str());
                    report(vector<string>(o.msgs.begin() + std::min<size_t>(1, o.msgs.size()), o.msgs.end()));
                });
        } else if (cmd == "merge" || cmd == "newmerge") {
            for (auto &fn : files) {
                HashTable o;
                o.load(fn);
                if (!ht.samplerate) ht.samplerate = o.samplerate;
                else if (o.samplerate && o.samplerate != ht.samplerate) throw std::runtime_error("samplerate mismatch in " + fn);
                ht.merge(o);
            }
        } else if (cmd == "list") {
            for (size_t i = 0; i < ht.names.size(); i++)
                if (!ht.names[i].empty()) report({ht.names[i] + " (" + std::to_string(ht.hashesperid[i]) + " hashes)"});
        } else if (cmd == "remove") {
            for (auto &fn : files) ht.remove(fn);
        }

        const double el = wall() - t0;
        if (totdur > 0.0)
            printf("Processed %d files (%.1f s total dur) in %.3f s sec = %.5f x RT\n", nprocessed, totdur, el, el / totdur);
        if (cmd != "precompute" && ht.dirty) ht.save(dbname);
        if (opf != stdout) fclose(opf);
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
