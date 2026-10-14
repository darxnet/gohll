package gohll

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// The binary encoding is the dense encoding of github.com/axiomhq/hyperloglog
// (version 2), so sketches can move between the two libraries:
//
//	byte 0    version (2)
//	byte 1    precision (14)
//	byte 2    0 (reserved)
//	byte 3    representation: 0 dense, 1 sparse
//	byte 4-7  number of registers, big-endian uint32 (16384)
//	byte 8-   one byte per register
//
// Only the dense representation is produced and accepted: a sparse axiomhq
// sketch must be converted (its Estimate does that past a size) before it is
// handed to UnmarshalBinary. A union of sketches from both libraries is only
// meaningful when both were fed the same 64-bit hashes.
const (
	encodingVersion = 2
	encodingDense   = 0
	encodingSparse  = 1
	headerLen       = 8
	encodedLen      = headerLen + registers
)

var (
	// ErrEncoding is wrapped by UnmarshalBinary for data it cannot read.
	ErrEncoding = errors.New("gohll: invalid encoding")
	// ErrSparseEncoding is wrapped by UnmarshalBinary for an axiomhq sketch in
	// its sparse representation.
	ErrSparseEncoding = errors.New("gohll: sparse encoding is not supported")
)

// MarshalBinary encodes the sketch; see the package's encoding note.
func (s *Sketch) MarshalBinary() ([]byte, error) {
	return s.AppendBinary(make([]byte, 0, encodedLen))
}

// AppendBinary appends the encoding of the sketch to data. Registers are read
// one by one, so Adds running concurrently may or may not be included.
func (s *Sketch) AppendBinary(data []byte) ([]byte, error) {
	data = append(data, encodingVersion, precision, 0, encodingDense)
	data = binary.BigEndian.AppendUint32(data, registers)

	for i := range s.words {
		cur := s.words[i].Load()

		for shift := 0; shift < 64; shift += 8 {
			data = append(data, byte(cur>>shift)) //nolint:gosec // G115 the low byte is the register
		}
	}

	return data, nil
}

// UnmarshalBinary replaces the sketch's registers with the encoded ones. It
// is not atomic with respect to concurrent Adds, like Reset.
func (s *Sketch) UnmarshalBinary(data []byte) error {
	if len(data) < headerLen {
		return fmt.Errorf("%w: %d bytes, header is %d", ErrEncoding, len(data), headerLen)
	}

	if data[0] != encodingVersion {
		return fmt.Errorf("%w: version %d, want %d", ErrEncoding, data[0], encodingVersion)
	}

	if data[1] != precision {
		return fmt.Errorf("%w: precision %d, want %d", ErrEncoding, data[1], precision)
	}

	switch data[3] {
	case encodingDense:
	case encodingSparse:
		return ErrSparseEncoding
	default:
		return fmt.Errorf("%w: representation %d", ErrEncoding, data[3])
	}

	if n := binary.BigEndian.Uint32(data[4:8]); n != registers {
		return fmt.Errorf("%w: %d registers, want %d", ErrEncoding, n, registers)
	}

	if len(data) != encodedLen {
		return fmt.Errorf("%w: %d bytes, want %d", ErrEncoding, len(data), encodedLen)
	}

	regs := data[headerLen:]

	for i := range s.words {
		var cur uint64

		for j := range registersPerWord {
			cur |= uint64(regs[i*registersPerWord+j]) << (8 * j)
		}

		s.words[i].Store(cur)
	}

	return nil
}
