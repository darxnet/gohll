# gohll

[![Go Reference](https://pkg.go.dev/badge/github.com/darxnet/gohll.svg)](https://pkg.go.dev/github.com/darxnet/gohll)
[![Go](https://github.com/darxnet/gohll/actions/workflows/release.yml/badge.svg)](https://github.com/darxnet/gohll/actions/workflows/release.yml)
[![Coverage](https://codecov.io/gh/darxnet/gohll/graph/badge.svg)](https://codecov.io/gh/darxnet/gohll)

`gohll` counts distinct items approximately with HyperLogLog — built for the
hot path: `Add` is lock-free, allocates nothing and takes a nanosecond, from
any number of goroutines at once.

## Why gohll?

A HyperLogLog sketch answers "how many distinct users / IPs / keys did we see"
in 16 KiB, whatever the number of items, with a standard error of 0.81%. The
Go libraries that implement it are not safe for concurrent use: every `Insert`
on a shared sketch needs a mutex, and on a server that counts on every request
the mutex costs more than the sketch.

`gohll` keeps the registers packed eight to a 64-bit word and updates them with
a compare-and-swap: no lock, no allocation, no coordination between goroutines.

| | `gohll` | `axiomhq/hyperloglog` |
|---|---|---|
| **Concurrent `Add`** | lock-free | mutex around `Insert` |
| **Memory per sketch** | 16 KiB, fixed, zero value ready | 16 KiB dense; smaller sparse mode for few items |
| **Estimate** | LogLog-Beta (HyperLogLog++ accuracy) | LogLog-Beta |
| **Merge, binary encoding** | yes; the encoding is axiomhq's dense format | yes |
| **Precision** | 14 | 4–18 |
| **Dependencies** | none | metro hash, intmap |

## Features

- **Lock-free `Add`** — one CAS on the register's word; ~1 ns on one
  goroutine, scales with cores.
- **Zero allocations** — `Add`, `Count`, `Merge`, `Reset`, `AppendBinary`.
- **HyperLogLog++ accuracy over the whole range** — the LogLog-Beta estimator
  ([Qin, Kim, Tung, Wang, 2016](https://arxiv.org/abs/1612.02284)) in one
  formula: tiny counts are exact or off by one, the 40K–80K band where plain
  HyperLogLog is biased is not, and the error is unbiased with the theoretical
  0.81% spread (tested over many seeds).
- **`Merge`** — the union of two sketches, usable while one is being written to.
- **Binary encoding compatible with `axiomhq/hyperloglog`** (dense, version 2):
  sketches can move between the two libraries and be merged if both were fed
  the same hashes. `encoding.BinaryMarshaler`, `BinaryUnmarshaler`,
  `BinaryAppender`.
- **You bring the hash** — `Add(hash uint64)`: xxhash, `hash/maphash`,
  whatever already hashed the item. The sketch does not hash again.
- **Zero dependencies** — only the standard library.

## Installation

```bash
go get github.com/darxnet/gohll
```

Requires Go 1.24.

## Quick Start

```go
// The zero value is ready to use; share it between goroutines as it is.
var users gohll.Sketch

// Add takes a 64-bit hash of the item.
users.Add(maphash.String(seed, userID))

// How many distinct items so far?
n := users.Count()

// Union with another sketch (same hash function on both sides).
users.Merge(&usersFromOtherServer)

// Persist, or send to another instance: axiomhq's dense encoding.
data, _ := users.MarshalBinary()
var back gohll.Sketch
_ = back.UnmarshalBinary(data)

// Start the next day.
users.Reset()
```

## Accuracy

Mean error and spread over 24 random data sets per size (`go test`, the
`TestSketch_Unbiased` gate):

| Distinct items | Mean error | Spread (1σ) |
|---|---|---|
| 1 – 100 | exact or ±1 | — |
| 20 000 | +0.08% | 0.62% |
| 40 000 | −0.06% | 0.67% |
| 60 000 | +0.11% | 0.65% |
| 80 000 | −0.06% | 0.63% |
| 120 000 | +0.05% | 0.72% |
| 300 000 | −0.04% | 0.79% |

Theory for 2^14 registers: 1.04/√16384 = 0.81%. Fed the same hashes,
`gohll.Count()` and `axiomhq.Estimate()` return the same number (both use
LogLog-Beta on identical dense registers).

## Benchmarks

> **Environment:** Apple M3 Pro · darwin/arm64 · Go 1.26
> `go test -bench=. -benchmem -count=5 -cpu=1,12 ./bench/`, summarised with `benchstat`.
> Same 64-bit hashes on both sides; `axiomhq` gets `InsertHash`, with and without the mutex a shared sketch needs.

| | `gohll` | `axiomhq` | `axiomhq` + mutex |
|---|---|---|---|
| **Add**, 1 goroutine | **1.08 ns** | 1.90 ns | 3.00 ns |
| **Add**, 12 goroutines on one sketch | **0.21 ns/op** | — | 117 ns/op |
| **Count / Estimate** (1M items) | **53 µs** | 171 µs | |
| **Allocs/op** | 0 | 0 | 0 |

Twelve goroutines adding to one sketch cost `axiomhq` 550x more per item than
`gohll`: that is the mutex, not the sketch. `bench/` is its own module, so the
library itself stays dependency-free.

## Design

- 2^14 registers of one byte, packed eight to an `atomic.Uint64`: `Add`
  extracts the register, compares, builds the new word and swaps it in; a lost
  race just retries. `Merge` takes the byte-wise maximum of each word the same way.
- `Count` reads every register once (16 KiB, L1-resident) and applies the
  LogLog-Beta formula; `Reset` and `UnmarshalBinary` store the words one by
  one, so they are not atomic with respect to concurrent `Add`s.
- No sparse representation on purpose: it would need a lock, and 16 KiB is
  the point of the fixed-size design. For many small sketches, prefer a
  library with sparse mode.

## See also

The sibling modules follow the same rule — nothing allocated on the hot path,
the zero value ready to use:

- [goshard](https://github.com/darxnet/goshard) — sharded, lock-based concurrent map for high-churn data: stores allocate nothing
- [gosync](https://github.com/darxnet/gosync) — generic concurrent map with `sync.Map` semantics and no interface boxing, for read-mostly data
- [goreset](https://github.com/darxnet/goreset) — generates `Reset()` for pooled structs: back to the zero state, memory kept
- [gocopy](https://github.com/darxnet/gocopy) — generates `CopyFrom`/`Copy` that reuse the destination's memory instead of allocating

## License

Apache License 2.0. See [LICENSE](LICENSE) for details. The LogLog-Beta
coefficients for precision 14 are the published fit from the paper, as also
used by `axiomhq/hyperloglog` (MIT).
