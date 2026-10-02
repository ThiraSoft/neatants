package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math/rand"
	"os"
	"testing"
)

// TestAimLearning follows the hit rate of network-aimed spells (fire, storm)
// in fresh full-brain worlds. Slow: NEATANTS_LONG_TESTS=1.
func TestAimLearning(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	wd, _ := os.Getwd()
	LoadConfig("config.yml")
	Cfg.BrainMode = "full"
	neat.Advanced = Cfg.Selection != "classic"
	os.Chdir(t.TempDir())
	defer os.Chdir(wd)
	ticks, seeds := 500000, 2
	if v := os.Getenv("NEATANTS_LEARN_TICKS"); v != "" {
		fmt.Sscan(v, &ticks)
	}
	const windows = 5
	for s := range seeds {
		rand.Seed(int64(500 + s))
		w := NewWorld()
		line := fmt.Sprintf("seed %d · hit rate of aimed shots per bucket:", s)
		lastShots, lastHits := 0, 0
		for tick := 1; tick <= ticks; tick++ {
			w.Update()
			w.Events = w.Events[:0]
			if tick%(ticks/windows) == 0 {
				shots, hits := w.AimShots-lastShots, w.AimHits-lastHits
				lastShots, lastHits = w.AimShots, w.AimHits
				line += fmt.Sprintf(" %.0f%% (%d shots)", 100*float64(hits)/float64(max(1, shots)), shots)
			}
		}
		fmt.Println(line)
	}
}
