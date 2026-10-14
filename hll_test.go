package gohll_test

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/darxnet/gohll"
)

// hashBytes is a deterministic stand-in for the caller's hash (FNV-1a 64
// finished with the splitmix64 mixer, so that every bit of the result is
// well spread): every run sees the same data and the accuracy gates do not flap.
func hashBytes(b []byte) uint64 {
	const (
		offset = 14695981039346656037
		prime  = 1099511628211
	)

	h := uint64(offset)
	for _, c := range b {
		h ^= uint64(c)
		h *= prime
	}

	h += 0x9E3779B97F4A7C15
	h ^= h >> 30
	h *= 0xBF58476D1CE4E5B9
	h ^= h >> 27
	h *= 0x94D049BB133111EB
	h ^= h >> 31

	return h
}

func TestSketch_Count(t *testing.T) {
	t.Parallel()

	// 40K–80K is where plain HyperLogLog is biased (between linear counting
	// and the raw estimate); LogLog-Beta must hold its error there too
	for _, n := range []int{0, 1, 10, 100, 1_000, 10_000, 40_000, 60_000, 80_000, 500_000, 5_000_000} {
		t.Run(fmtInt(n), func(t *testing.T) {
			t.Parallel()

			var (
				s   gohll.Sketch
				buf [8]byte
				rng = rand.New(rand.NewPCG(uint64(n), 7)) //nolint:gosec // deterministic test data
			)

			for range n {
				binary.LittleEndian.PutUint64(buf[:], rng.Uint64())
				s.Add(hashBytes(buf[:]))
			}

			got := float64(s.Count())
			// one sample: three standard errors (0.81% each); the bias check
			// with the tight bound is TestSketch_Unbiased
			tolerance := 0.025 * float64(n)

			if n <= 100 {
				tolerance = 1 // tiny counts are exact or off by one
			}

			if math.Abs(got-float64(n)) > tolerance {
				t.Fatalf("Count() = %.0f for %d (%.2f) distinct items, want within %.0f", got, n, 100-100*got/float64(n), tolerance)
			}

			if (n == 0) != s.Empty() {
				t.Fatalf("Empty() = %v for %d items", s.Empty(), n)
			}
		})
	}
}

// TestSketch_Unbiased is the HyperLogLog++ property: over many samples the
// error averages out to zero across the range — including the 40K–80K band
// where plain HyperLogLog is biased — with the spread of the theory.
func TestSketch_Unbiased(t *testing.T) {
	t.Parallel()

	const seeds = 24

	for _, n := range []int{20_000, 40_000, 60_000, 80_000, 120_000} {
		t.Run(fmtInt(n), func(t *testing.T) {
			t.Parallel()

			var sum, sumSq float64

			for seed := range uint64(seeds) {
				var (
					s   gohll.Sketch
					buf [8]byte
					rng = rand.New(rand.NewPCG(seed*7919+1, uint64(n))) //nolint:gosec // deterministic test data
				)

				for range n {
					binary.LittleEndian.PutUint64(buf[:], rng.Uint64())
					s.Add(hashBytes(buf[:]))
				}

				e := (float64(s.Count()) - float64(n)) / float64(n)
				sum += e
				sumSq += e * e
			}

			mean := sum / seeds
			std := math.Sqrt(sumSq/seeds - mean*mean)

			if math.Abs(mean) > 0.004 {
				t.Fatalf("mean error %+.2f%% over %d samples, want |bias| below 0.4%%", mean*100, seeds)
			}

			if std > 0.011 {
				t.Fatalf("error spread %.2f%%, want at most 1.1%% (theory 0.81%%)", std*100)
			}
		})
	}
}

func TestSketch_Merge(t *testing.T) {
	t.Parallel()

	var a, b, both gohll.Sketch

	add := func(s *gohll.Sketch, from, to int) {
		var buf [8]byte

		for i := from; i < to; i++ {
			binary.LittleEndian.PutUint64(buf[:], uint64(i)) //nolint:gosec // G115 test data
			s.Add(hashBytes(buf[:]))
		}
	}

	add(&a, 0, 60_000)
	add(&b, 40_000, 100_000) // 20_000 shared with a
	add(&both, 0, 100_000)

	a.Merge(&b)

	want := both.Count()

	if got := a.Count(); got != want {
		t.Fatalf("Count() after Merge = %d, want %d: the union sketch must equal the sketch of the union", got, want)
	}

	var empty gohll.Sketch

	a.Merge(&empty)

	if got := a.Count(); got != want {
		t.Fatalf("Merge of an empty sketch changed the count to %d", got)
	}

	empty.Merge(&b)

	if got, want := empty.Count(), b.Count(); got != want {
		t.Fatalf("Merge into an empty sketch = %d, want %d", got, want)
	}
}

func TestSketch_Duplicates(t *testing.T) {
	t.Parallel()

	var s gohll.Sketch

	for range 1_000_000 {
		s.Add(hashBytes([]byte("the same user")))
	}

	if got := s.Count(); got != 1 {
		t.Fatalf("Count() = %d for one distinct item added a million times", got)
	}
}

func TestSketch_ConcurrentAddAndReset(t *testing.T) {
	t.Parallel()

	var (
		s  gohll.Sketch
		wg sync.WaitGroup
	)

	const perGoroutine = 20_000

	for g := range uint64(8) {
		wg.Add(1)

		go func() {
			defer wg.Done()

			var buf [8]byte

			for i := range uint64(perGoroutine) {
				binary.LittleEndian.PutUint64(buf[:], g<<32|i)
				s.Add(hashBytes(buf[:]))
			}
		}()
	}

	wg.Wait()

	const n = 8 * perGoroutine
	if got := float64(s.Count()); math.Abs(got-n) > 0.03*n {
		t.Fatalf("Count() = %.0f after concurrent adds of %d items", got, n)
	}

	s.Reset()

	if !s.Empty() || s.Count() != 0 {
		t.Fatalf("after Reset: Empty() = %v, Count() = %d", s.Empty(), s.Count())
	}
}

func BenchmarkSketch_Add(b *testing.B) {
	var s gohll.Sketch

	b.ReportAllocs()

	for i := range b.N {
		s.Add(uint64(i) * 0x9E3779B97F4A7C15)
	}
}

func BenchmarkSketch_Count(b *testing.B) {
	var s gohll.Sketch

	for i := range 1_000_000 {
		s.Add(uint64(i) * 0x9E3779B97F4A7C15)
	}

	b.ReportAllocs()

	for b.Loop() {
		_ = s.Count()
	}
}

func fmtInt(n int) string {
	switch {
	case n >= 1_000_000:
		return "n=" + itoa(n/1_000_000) + "M"
	case n >= 1_000:
		return "n=" + itoa(n/1_000) + "K"
	default:
		return "n=" + itoa(n)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}

	var b [20]byte

	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}

	return string(b[i:])
}

func BenchmarkSketch_AddParallel(b *testing.B) {
	var s gohll.Sketch

	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		i := uint64(rand.Uint32()) //nolint:gosec // benchmark
		for pb.Next() {
			i++
			s.Add(i * 0x9E3779B97F4A7C15)
		}
	})
}
