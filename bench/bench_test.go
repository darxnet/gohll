// Package bench compares gohll with github.com/axiomhq/hyperloglog on the
// same hashes: Add on one goroutine and on all of them (axiomhq needs a
// mutex), and the estimate.
package bench

import (
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

// TestSameEstimate feeds both the same hashes: dense-mode estimates must agree
// closely (both use LogLog-Beta on identical registers).
func TestSameEstimate(t *testing.T) {
	var s gohll.Sketch

	a := hyperloglog.New14()

	for i := range uint64(500_000) {
		h := i * 0x9E3779B97F4A7C15
		s.Add(h)
		a.InsertHash(h)
	}

	ours, theirs := s.Count(), a.Estimate()
	t.Logf("gohll %d, axiomhq %d", ours, theirs)

	if diff := float64(ours) - float64(theirs); diff > 0.001*float64(theirs) || diff < -0.001*float64(theirs) {
		t.Fatalf("estimates differ by more than 0.1%%: %d vs %d", ours, theirs)
	}
}
