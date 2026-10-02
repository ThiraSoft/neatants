package sim

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestRefoundDiversity compares element diversity for each refound source.
// Slow: runs only with NEATANTS_LONG_TESTS=1.
func TestRefoundDiversity(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	wd, _ := os.Getwd()
	for _, mode := range []string{"best", "own", "fresh"} {
		os.Chdir(wd)
		LoadConfig("config.yml")
		Cfg.RefoundFrom = mode
		os.Chdir(t.TempDir())
		entropy, top, samples := 0.0, 0.0, 0
		for seed := range 3 {
			rand.Seed(int64(100 + seed))
			w := NewWorld()
			for tick := 1; tick <= 120000; tick++ {
				w.Update()
				w.Events = w.Events[:0]
				if tick%5000 != 0 || tick < 20000 {
					continue
				}
				var cnt [NumElements]float64
				tot := 0.0
				for _, a := range w.Ants {
					if a.Alive {
						cnt[a.Element]++
						tot++
					}
				}
				h, mx := 0.0, 0.0
				for _, n := range cnt {
					if n > 0 {
						p := n / tot
						h -= p * math.Log(p)
						mx = math.Max(mx, p)
					}
				}
				entropy += h / math.Log(NumElements)
				top += mx
				samples++
			}
		}
		fmt.Printf("refounding %-5s: diversity %.2f (1 = the 5 elements evenly split) · dominant element on average %.0f %% of ants\n",
			mode, entropy/float64(samples), 100*top/float64(samples))
	}
	os.Chdir(wd)
}
