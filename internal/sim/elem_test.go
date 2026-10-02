package sim

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// TestElementDiversity follows how elements spread in fresh worlds.
// Slow: runs only with NEATANTS_LONG_TESTS=1.
func TestElementDiversity(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	LoadConfig("config.yml")
	wd, _ := os.Getwd()
	os.Chdir(t.TempDir()) // no save: fresh lineages
	defer os.Chdir(wd)
	for seed := range 3 {
		rand.Seed(int64(100 + seed))
		w := NewWorld()
		var fitSum [NumElements]float64
		var fitN [NumElements]int
		refounds := 0
		for tick := 1; tick <= 120000; tick++ {
			w.Update()
			for _, e := range w.Events {
				if e.Kind == EvColonyFound && tick > 1 {
					refounds++
				}
			}
			w.Events = w.Events[:0]
			for _, a := range w.Ants {
				if !a.Alive && a.Age > 0 && a.MaxAge > 0 {
					fit := float64(a.Delivered)*3 + float64(a.Kills)*2 + float64(a.MonsterKills)*5 + a.Healed/8 + float64(a.Age)/6000
					fitSum[a.Element] += fit
					fitN[a.Element]++
					a.MaxAge = 0 // count once
				}
			}
			if tick%20000 == 0 {
				var cnt [NumElements]int
				tot := 0
				per := ""
				for _, c := range w.Colonies {
					dom, dn := 0, -1
					for e, n := range c.ElemCount {
						if n > dn {
							dom, dn = e, n
						}
					}
					if c.Alive {
						per += fmt.Sprintf(" %s:%s", c.Name, ElemNames[dom])
					}
				}
				for _, a := range w.Ants {
					if a.Alive {
						cnt[a.Element]++
						tot++
					}
				}
				line := ""
				for e := range NumElements {
					line += fmt.Sprintf(" %s %2.0f%%", ElemNames[e], 100*float64(cnt[e])/float64(max(1, tot)))
				}
				fmt.Printf("seed %d t=%6d |%s | dominant:%s\n", seed, tick, line, per)
			}
		}
		score := ""
		for e := range NumElements {
			score += fmt.Sprintf(" %s %.2f (%d)", ElemNames[e], fitSum[e]/float64(max(1, fitN[e])), fitN[e])
		}
		fmt.Printf("seed %d: refoundings %d · mean score at death:%s\n", seed, refounds, score)
	}
}
