// Package gohll counts distinct items approximately with HyperLogLog.
//
// A Sketch keeps m = 2^14 one-byte registers (16 KiB, eight to a 64-bit word
// for lock-free updates) whatever the number of items, and estimates their
// cardinality with a standard error of 1.04/sqrt(m) ≈ 0.8%. Items are added
// as 64-bit hashes: the top 14 bits pick a register, the register keeps the
// longest run of leading zeros seen in the rest, plus one.
//
// The estimate is the LogLog-Beta one (Qin, Kim, Tung, Wang, 2016): a single
// formula with a per-precision polynomial correction that covers the whole
// range, from a handful of items to billions, with the accuracy of the
// HyperLogLog++ bias tables and without its switch between estimators.
//
// Add is lock-free (a compare-and-swap max on the register's word) and
// allocates nothing; Count walks the registers; Merge takes the union of two
// sketches; Reset starts over.
package gohll

import (
	"math"
	"math/bits"
	"sync/atomic"
)

const (
	precision = 14
	registers = 1 << precision // m

	// registers are bytes packed eight to a word: register i is byte i&7 of
	// word i>>3, the low byte first.
	registersPerWord = 8
	words            = registers / registersPerWord
	registerMask     = 0xFF

	// alpha is the bias correction of the raw estimate for m >= 128.
	alpha = 0.7213 / (1 + 1.079/registers)
)

// Sketch is a HyperLogLog counter. The zero value is ready to use.
type Sketch struct {
	words [words]atomic.Uint64
}

// Add records one item given by its 64-bit hash. The hash must spread well
// over all bits (xxhash, maphash...): the sketch does not hash again.
func (s *Sketch) Add(hash uint64) {
	i := hash >> (64 - precision)
	// the rank: leading zeros of the remaining 50 bits, plus one; the top 14
	// bits are forced to one so that they do not count
	rank := uint64(bits.LeadingZeros64(hash<<precision|(1<<precision-1))) + 1 //nolint:gosec // G115 at most 51

	w := &s.words[i/registersPerWord]
	shift := (i % registersPerWord) * 8

	for {
		cur := w.Load()
		if (cur>>shift)&registerMask >= rank {
			return
		}

		next := cur&^(registerMask<<shift) | rank<<shift
		if w.CompareAndSwap(cur, next) {
			return
		}
	}
}

// Count estimates the number of distinct items added since the last Reset.
func (s *Sketch) Count() uint64 {
	var (
		sum   float64 // Σ 2^-register
		empty float64 // registers still at zero
	)

	for i := range s.words {
		cur := s.words[i].Load()

		for shift := 0; shift < 64; shift += 8 {
			v := (cur >> shift) & registerMask
			sum += math.Ldexp(1, -int(v)) //nolint:gosec // G115 at most 51

			if v == 0 {
				empty++
			}
		}
	}

	if empty == registers {
		return 0
	}

	// LogLog-Beta: the raw estimate over the non-empty registers, with the
	// polynomial in the number of empty ones standing in for both the
	// small-range (linear counting) and the mid-range bias corrections.
	estimate := alpha * registers * (registers - empty) / (sum + beta(empty))

	return uint64(estimate + 0.5)
}

// beta is the LogLog-Beta correction for precision 14: the coefficients come
// from the paper's fit (table for p=14), as used by axiomhq/hyperloglog (MIT).
func beta(empty float64) float64 {
	zl := math.Log(empty + 1)
	zl2 := zl * zl
	zl3 := zl2 * zl
	zl4 := zl3 * zl

	return -0.371009760230692*empty +
		0.00978811941207509*zl +
		0.185796293324165*zl2 +
		0.203015527328432*zl3 +
		-0.116710521803686*zl4 +
		0.0431106699492820*zl4*zl +
		-0.00599583540511831*zl4*zl2 +
		0.000449704299509437*zl4*zl3
}

// Empty reports whether nothing was added since the last Reset.
func (s *Sketch) Empty() bool {
	for i := range s.words {
		if s.words[i].Load() != 0 {
			return false
		}
	}

	return true
}

// Reset forgets every item. It is not atomic with respect to concurrent Adds:
// an Add racing with Reset may land before or after it.
func (s *Sketch) Reset() {
	for i := range s.words {
		s.words[i].Store(0)
	}
}

// Merge adds every item of other to s: the union of the two sets, exact in
// the sense that it is the sketch the union would have produced. Both must
// have been fed the same hash function. other is only read; s may be in use
// by concurrent Adds.
func (s *Sketch) Merge(other *Sketch) {
	for i := range s.words {
		o := other.words[i].Load()
		if o == 0 {
			continue
		}

		w := &s.words[i]

		for {
			cur := w.Load()

			next := maxBytes(cur, o)
			if next == cur || w.CompareAndSwap(cur, next) {
				break
			}
		}
	}
}

// maxBytes takes the larger of each of the eight bytes of a and b.
func maxBytes(a, b uint64) uint64 {
	var out uint64

	for shift := 0; shift < 64; shift += 8 {
		out |= max((a>>shift)&registerMask, (b>>shift)&registerMask) << shift
	}

	return out
}
