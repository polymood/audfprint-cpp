CXXFLAGS ?= -O3 -march=native
# -ffp-contract=off: no FMA contraction, keeps float math close to numpy/scipy
CXXFLAGS += -std=c++20 -ffp-contract=off -Wall -Ithird_party/ffsrc
LDLIBS = -l:libavformat.so.62 -l:libavcodec.so.62 -l:libswresample.so.6 -l:libavutil.so.60 -lmvec -lz -pthread

audfprint: audfprint.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
