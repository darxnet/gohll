package gohll_test

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"testing"

	"github.com/darxnet/gohll"
)

var (
	_ encoding.BinaryMarshaler   = (*gohll.Sketch)(nil)
	_ encoding.BinaryUnmarshaler = (*gohll.Sketch)(nil)
	_ encoding.BinaryAppender    = (*gohll.Sketch)(nil)
)

// axiomhqDense is what github.com/axiomhq/hyperloglog v0.3.0 writes for a
// dense precision-14 sketch: the header, then one byte per register.
func axiomhqDense(regs []byte) []byte {
	data := []byte{2, 14, 0, 0}
	data = binary.BigEndian.AppendUint32(data, 1<<14)

	return append(data, regs...)
}

func TestSketch_BinaryRoundTrip(t *testing.T) {
	t.Parallel()

	var s gohll.Sketch

	for i := range 100_000 {
		var buf [8]byte

		binary.LittleEndian.PutUint64(buf[:], uint64(i))
		s.Add(hashBytes(buf[:]))
	}

	data, err := s.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	if len(data) != 8+1<<14 {
		t.Fatalf("len = %d, want header + one byte per register", len(data))
	}

	if !bytes.Equal(data[:8], axiomhqDense(nil)) {
		t.Fatalf("header = %v, want the axiomhq dense header", data[:8])
	}

	var back gohll.Sketch

	if err = back.UnmarshalBinary(data); err != nil {
		t.Fatal(err)
	}

	if got, want := back.Count(), s.Count(); got != want {
		t.Fatalf("Count() after round trip = %d, want %d", got, want)
	}

	again, err := back.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(again, data) {
		t.Fatal("the encoding must be canonical for the dense representation")
	}
}

func TestSketch_AppendBinary(t *testing.T) {
	t.Parallel()

	var s gohll.Sketch

	prefix := []byte("frame:")

	data, err := s.AppendBinary(prefix)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.HasPrefix(data, prefix) || len(data) != len(prefix)+8+1<<14 {
		t.Fatalf("AppendBinary must append after the caller's %d bytes, got %d", len(prefix), len(data))
	}

	if !bytes.Equal(data[len(prefix):], axiomhqDense(make([]byte, 1<<14))) {
		t.Fatal("an empty sketch encodes as all-zero registers")
	}
}

func TestSketch_UnmarshalBinaryFromAxiomhq(t *testing.T) {
	t.Parallel()

	// a dense sketch as the other library would write it: register 7 at rank 3
	regs := make([]byte, 1<<14)
	regs[7] = 3

	var s gohll.Sketch

	if err := s.UnmarshalBinary(axiomhqDense(regs)); err != nil {
		t.Fatal(err)
	}

	if s.Empty() {
		t.Fatal("the decoded sketch must hold the register")
	}

	if got := s.Count(); got != 1 {
		t.Fatalf("Count() = %d, want 1 for one touched register", got)
	}
}

func TestSketch_UnmarshalBinaryErrors(t *testing.T) {
	t.Parallel()

	regs := make([]byte, 1<<14)
	good := axiomhqDense(regs)

	tests := []struct {
		name string
		data []byte
		want error
	}{
		{name: "short", data: good[:5], want: gohll.ErrEncoding},
		{name: "version", data: append([]byte{1}, good[1:]...), want: gohll.ErrEncoding},
		{name: "precision", data: append([]byte{2, 12}, good[2:]...), want: gohll.ErrEncoding},
		{name: "sparse", data: append([]byte{2, 14, 0, 1}, good[4:]...), want: gohll.ErrSparseEncoding},
		{name: "representation", data: append([]byte{2, 14, 0, 9}, good[4:]...), want: gohll.ErrEncoding},
		{
			name: "registers",
			data: append(binary.BigEndian.AppendUint32([]byte{2, 14, 0, 0}, 4096), regs...),
			want: gohll.ErrEncoding,
		},
		{name: "trailing", data: append(append([]byte{}, good...), 0), want: gohll.ErrEncoding},
		{name: "truncated", data: good[:len(good)-1], want: gohll.ErrEncoding},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var s gohll.Sketch

			if err := s.UnmarshalBinary(tt.data); !errors.Is(err, tt.want) {
				t.Fatalf("UnmarshalBinary error = %v, want %v", err, tt.want)
			}
		})
	}
}
