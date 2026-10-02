package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// TestLifeCycle measures lifespans, causes of death, gene transmission and
// colony collapses. Slow: runs only with NEATANTS_LONG_TESTS=1.
func TestLifeCycle(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	LoadConfig("config.yml")
	if v := os.Getenv("NEATANTS_SELECTION"); v != "" {
		Cfg.Selection = v
		neat.Advanced = v != "classic"
	}
	wd, _ := os.Getwd()
	os.Chdir(t.TempDir())
	defer os.Chdir(wd)
	seed := int64(7)
	if v := os.Getenv("NEATANTS_SEED"); v != "" {
		fmt.Sscan(v, &seed)
	}
	rand.Seed(seed)
	w := NewWorld()
	var ages []int
	causes := map[string]int{}
	transmitted, died := 0, 0
	falls := map[string]int{}
	poolSamples := map[int]int{}
	lastPop := map[int]int{}
	for tick := 1; tick <= 100000; tick++ {
		for _, c := range w.Colonies {
			lastPop[c.ID] = c.Pop
		}
		prevRecent := map[int]int{}
		for _, c := range w.Colonies {
			prevRecent[c.ID] = len(c.Recent)
		}
		alive := map[*Ant]bool{}
		for _, a := range w.Ants {
			if a.Alive {
				alive[a] = true
			}
		}
		foodBefore := map[int]float64{}
		for _, c := range w.Colonies {
			foodBefore[c.ID] = c.Food
		}
		w.Update()
		for _, e := range w.Events {
			if e.Kind == EvNestMove {
				falls["relocation (survives)"]++
			}
			if e.Kind == EvColonyFall {
				c := w.Colonies[e.Colony]
				if c.HP <= 0 {
					falls["nest destroyed"]++
				} else {
					falls[fmt.Sprintf("famine (stock %.0f)", foodBefore[e.Colony])]++
				}
			}
		}
		w.Events = w.Events[:0]
		for a := range alive {
			if a.Alive {
				continue
			}
			died++
			ages = append(ages, a.Age)
			switch {
			case a.HP <= 0:
				causes["killed"]++
			case a.Age > a.MaxAge:
				causes["old age"]++
			default:
				causes["hunger"]++
			}
			fit := float64(a.Delivered)*3 + float64(a.Kills)*2 + float64(a.MonsterKills)*5 + a.Impact/10 + float64(a.Age)/6000
			if fit > 0.5 {
				transmitted++
			}
		}
		if tick%10000 == 0 {
			for _, c := range w.Colonies {
				if c.Alive {
					poolSamples[len(c.Hall)+len(c.Recent)]++
				}
			}
		}
	}
	sort.Ints(ages)
	pct := func(p float64) int { return ages[int(p*float64(len(ages)-1))] }
	fmt.Printf("deaths %d · age at death (ticks): median %d, 25 %% %d, 75 %% %d, 90 %% %d · max lifespan %d to %d\n",
		died, pct(0.5), pct(0.25), pct(0.75), pct(0.9), Cfg.AntMaxAge/2, Cfg.AntMaxAge)
	fmt.Printf("causes: %v\n", causes)
	fmt.Printf("genes passed on (score > 0.5): %d of %d (%.0f %%)\n", transmitted, died, 100*float64(transmitted)/float64(max(1, died)))
	fmt.Printf("collapses: %v\n", falls)
	fmt.Printf("gene pool size of living colonies (samples): %v\n", poolSamples)
}
