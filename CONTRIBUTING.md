# Contributing

Issues and pull requests are welcome.

## Build and test

```sh
make build        # game and headless binaries
make test         # go vet + fast tests
make test-long    # also runs the slow tests (diversity, marriage, learning...)
```

The slow tests are skipped unless `NEATANTS_LONG_TESTS=1` is set. `make test-long` sets it for you and can take a long time.

## Guidelines

- Keep `neat/` and `internal/sim/` free of Ebiten imports. Everything that draws lives in `internal/render/`. The headless runner must build without a display or the Ebiten system libraries.
- Run `gofmt -l .` and `go vet ./...` before opening a PR, and make sure `go test ./...` passes.
- Add a test when you change simulation behaviour. Prefer small deterministic checks over long runs.
- Keep changes focused. One topic per PR.
