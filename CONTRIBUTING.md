# Contributing

`gohll` is maintained by one person; issues and pull requests are welcome.
Please open an issue first for anything bigger than a bug fix, so the design is
agreed before the code is written.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org) are required: the
release version is computed from them by [`svu`](https://github.com/caarlos0/svu)
in CI when a commit lands on `main`.

| Prefix | Release |
|---|---|
| `fix:` | patch |
| `feat:` | minor |
| `feat!:` or a `BREAKING CHANGE:` footer | major |
| `chore:`, `docs:`, `test:`, `ci:`, `refactor:`, `perf:` | none |

A `perf:` change that users should know about belongs in a `feat:` commit.

## Before a pull request

    go mod tidy
    go vet ./...
    go test -race -count=1 ./...
    golangci-lint run ./...                # .golangci.yml, `default: all`

CI runs the same checks on the stable and oldstable Go releases, on Linux (amd64) and macOS (arm64),
plus `govulncheck` and a coverage gate.

## Changes on a hot path

Include `benchstat` output in the pull request description:

    go test -run=NONE -bench=. -benchmem -count=6 ./... > old.txt   # on main
    go test -run=NONE -bench=. -benchmem -count=6 ./... > new.txt   # on your branch
    benchstat old.txt new.txt

A change that makes a benchmark slower needs a reason in the commit message.

## Ground rules

- Every behaviour change comes with a test; coverage is gated in CI.
- No new dependencies: the module deliberately uses only the standard library.
- The zero value of every exported type must stay ready to use.
- `bench/` is a separate module (it imports axiomhq/hyperloglog for comparison): the
  library itself must stay dependency-free. Refresh the README table from it when a
  hot path changes.
