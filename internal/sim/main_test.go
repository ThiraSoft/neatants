package sim

import (
	"os"
	"testing"
)

// TestMain runs the tests from the repository root, where config.yml and the
// saves/ directory live.
func TestMain(m *testing.M) {
	if err := os.Chdir("../.."); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}
