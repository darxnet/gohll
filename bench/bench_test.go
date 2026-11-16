// Package bench compares gohll with github.com/axiomhq/hyperloglog on the
// same hashes: Add on one goroutine and on all of them (axiomhq needs a
// mutex), and the estimate.
package bench

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"sync"
	"testing"
	"unsafe"

	"github.com/axiomhq/hyperloglog"

	"github.com/darxnet/gohll"
)

const prefill = 1_000_000

func fill(add func(uint64)) {
	for i := range uint64(prefill) {
		add(i * 0x9E3779B97F4A7C15)
	}
}

func BenchmarkAdd(b *testing.B) {
	b.Run("impl=gohll", func(b *testing.B) {
		var s gohll.Sketch

		b.ReportAllocs()

		for i := range b.N {
			s.Add(uint64(i) * 0x9E3779B97F4A7C15) //nolint:gosec // G115
		}
	})

	b.Run("impl=axiomhq", func(b *testing.B) {
		s := hyperloglog.New14()

		b.ReportAllocs()

		for i := range b.N {
			s.InsertHash(uint64(i) * 0x9E3779B97F4A7C15) //nolint:gosec // G115
		}
	})

	b.Run("impl=axiomhq+mutex", func(b *testing.B) {
		s := hyperloglog.New14()

		var mu sync.Mutex

		b.ReportAllocs()

		for i := range b.N {
			mu.Lock()
			s.InsertHash(uint64(i) * 0x9E3779B97F4A7C15) //nolint:gosec // G115
			mu.Unlock()
		}
	})
}

func BenchmarkAddParallel(b *testing.B) {
	b.Run("impl=gohll", func(b *testing.B) {
		var s gohll.Sketch

		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			i := uint64(rand.Uint32()) //nolint:gosec // benchmark
			for pb.Next() {
				i++
				s.Add(i * 0x9E3779B97F4A7C15)
			}
		})
	})

	b.Run("impl=axiomhq+mutex", func(b *testing.B) {
		s := hyperloglog.New14()

		var mu sync.Mutex

		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			i := uint64(rand.Uint32()) //nolint:gosec // benchmark
			for pb.Next() {
				i++
				mu.Lock()
				s.InsertHash(i * 0x9E3779B97F4A7C15)
				mu.Unlock()
			}
		})
	})
}

func BenchmarkCount(b *testing.B) {
	b.Run("impl=gohll", func(b *testing.B) {
		var s gohll.Sketch

		fill(s.Add)
		b.ReportAllocs()

		for b.Loop() {
			_ = s.Count()
		}
	})

	b.Run("impl=axiomhq", func(b *testing.B) {
		s := hyperloglog.New14()

		fill(s.InsertHash)
		b.ReportAllocs()

		for b.Loop() {
			_ = s.Estimate()
		}
	})
}

func TestSizes(t *testing.T) {
	var s gohll.Sketch

	t.Logf("gohll.Sketch: %d bytes in memory", unsafe.Sizeof(s))

	a := hyperloglog.New14()
	fill(a.InsertHash)

	data, _ := a.MarshalBinary()
	t.Logf("axiomhq dense: %d bytes encoded (16 KiB registers + header)", len(data))

	ours, _ := s.MarshalBinary()
	t.Logf("gohll encoded: %d bytes", len(ours))
}

// TestSameEstimate feeds both the same hashes. Against an axiomhq sketch that
// is dense from the start (NewNoSparse) the registers are identical and both
// apply LogLog-Beta, so the estimates must agree at every size. New14 starts
// sparse and estimates by linear counting at precision 25 until it converts,
// which is not compared here; past that point it is checked too.
func TestSameEstimate(t *testing.T) {
	t.Parallel()

	var s gohll.Sketch

	dense := hyperloglog.NewNoSparse()
	sparse := hyperloglog.New14()

	check := func(name string, ours, theirs uint64) {
		t.Helper()

		// identical registers: allow one for floating-point rounding (FMA)
		if diff := int64(ours) - int64(theirs); diff > 1 || diff < -1 { //nolint:gosec // G115 small counts
			t.Fatalf("%s: gohll %d, axiomhq %d", name, ours, theirs)
		}
	}

	var n uint64

	for _, size := range []uint64{1, 10, 100, 1_000, 10_000, 100_000, 500_000} {
		for ; n < size; n++ {
			h := mix(n)
			s.Add(h)
			dense.InsertHash(h)
			sparse.InsertHash(h)
		}

		check("NewNoSparse", s.Count(), dense.Estimate())
	}

	// long converted to dense by now
	check("New14", s.Count(), sparse.Estimate())
}

// FuzzSameEstimate feeds two gohll sketches and two dense axiomhq sketches
// the same hashes (8 bytes of input each), then merges the second into the
// first on both sides. After every step the encodings must be byte-identical
// (same registers, same header) and the estimates must agree; an axiomhq
// encoding must also load into gohll and keep its estimate.
func FuzzSameEstimate(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add(hashes(0), hashes(^uint64(0)))
	f.Add(hashes(1, 1<<63, 1<<50), hashes(1<<49, 1<<50+1))

	big := make([]uint64, 5000)
	for i := range big {
		big[i] = mix(uint64(i)) //nolint:gosec // G115
	}

	f.Add(hashes(big[:3000]...), hashes(big[2000:]...))

	f.Fuzz(func(t *testing.T, a, b []byte) {
		var ours, other gohll.Sketch

		theirs := hyperloglog.NewNoSparse()
		theirOther := hyperloglog.NewNoSparse()

		for h := range split(a) {
			ours.Add(h)
			theirs.InsertHash(h)
		}

		for h := range split(b) {
			other.Add(h)
			theirOther.InsertHash(h)
		}

		same(t, "a", &ours, theirs)
		same(t, "b", &other, theirOther)

		ours.Merge(&other)

		if err := theirs.Merge(theirOther); err != nil {
			t.Fatal(err)
		}

		same(t, "a∪b", &ours, theirs)
	})
}

// same checks that s and a hold the same registers and estimate alike, and
// that a's encoding loads into gohll.
func same(t *testing.T, name string, s *gohll.Sketch, a *hyperloglog.Sketch) {
	t.Helper()

	ours, err := s.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	theirs, err := a.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(ours, theirs) {
		t.Fatalf("%s: gohll and axiomhq encodings differ", name)
	}

	// identical registers: allow one for floating-point rounding (FMA)
	got, want := s.Count(), a.Estimate()
	if diff := int64(got) - int64(want); diff > 1 || diff < -1 { //nolint:gosec // G115 counts fit
		t.Fatalf("%s: gohll %d, axiomhq %d", name, got, want)
	}

	var loaded gohll.Sketch
	if err = loaded.UnmarshalBinary(theirs); err != nil {
		t.Fatalf("%s: UnmarshalBinary(axiomhq): %v", name, err)
	}

	if c := loaded.Count(); c != got {
		t.Fatalf("%s: loaded from axiomhq %d, built directly %d", name, c, got)
	}
}

// hashes encodes hs as fuzz input, 8 bytes each.
func hashes(hs ...uint64) []byte {
	out := make([]byte, 0, 8*len(hs))
	for _, h := range hs {
		out = binary.LittleEndian.AppendUint64(out, h)
	}

	return out
}

// split yields the input as 64-bit hashes, ignoring a trailing partial one.
func split(data []byte) func(func(uint64) bool) {
	return func(yield func(uint64) bool) {
		for ; len(data) >= 8; data = data[8:] {
			if !yield(binary.LittleEndian.Uint64(data)) {
				return
			}
		}
	}
}

// mix is the SplitMix64 finalizer: a well-spread hash of n.
func mix(n uint64) uint64 {
	n += 0x9E3779B97F4A7C15
	n ^= n >> 30
	n *= 0xBF58476D1CE4E5B9
	n ^= n >> 27
	n *= 0x94D049BB133111EB

	return n ^ n>>31
}
