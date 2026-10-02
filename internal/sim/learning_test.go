package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"math/rand"
	"os"
	"testing"
)

// learningRun plays a fresh full-brain world and measures, from the ants
// that died in two windows, how much food and how many monsters each ant
// handled per 10k ticks of life (robust to colony refounds).
func learningRun(seed int64, ticks int) (early, late [2]float64) {
	rand.Seed(seed)
	w := NewWorld()
	var sums [2][3]float64 // window -> delivered, monster kills, age
	alive := map[*Ant]bool{}
	for tick := 1; tick <= ticks; tick++ {
		w.Update()
		w.Events = w.Events[:0]
		win := -1
		switch {
		case tick > ticks/12 && tick <= ticks/4:
			win = 0
		case tick > ticks*3/4:
			win = 1
		}
		for _, a := range w.Ants {
			if a.Alive {
				alive[a] = true
			}
		}
		for a := range alive {
			if !a.Alive {
				delete(alive, a)
				if win >= 0 {
					sums[win][0] += float64(a.Delivered)
					sums[win][1] += float64(a.MonsterKills)
					sums[win][2] += float64(a.Age)
				}
			}
		}
	}
	for i := range 2 {
		f := 10000 / max(1, sums[i][2])
		v := [2]float64{sums[i][0] * f, sums[i][1] * f}
		if i == 0 {
			early = v
		} else {
			late = v
		}
	}
	return
}

// TestSelectionLearning compares classic and advanced selection on how fast
// brains alone learn to forage and fight. Slow: NEATANTS_LONG_TESTS=1.
func TestSelectionLearning(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	wd, _ := os.Getwd()
	ticks, seeds := 120000, 4
	if v := os.Getenv("NEATANTS_LEARN_TICKS"); v != "" {
		fmt.Sscan(v, &ticks)
	}
	if v := os.Getenv("NEATANTS_LEARN_SEEDS"); v != "" {
		fmt.Sscan(v, &seeds)
	}
	for _, mode := range []string{"classic", "advanced"} {
		os.Chdir(wd)
		LoadConfig("config.yml")
		Cfg.BrainMode, Cfg.Selection = "full", mode
		neat.Advanced = mode == "advanced"
		os.Chdir(t.TempDir())
		var e, l [2]float64
		for s := range seeds {
			ee, ll := learningRun(int64(300+s), ticks)
			for i := range 2 {
				e[i] += ee[i] / float64(seeds)
				l[i] += ll[i] / float64(seeds)
			}
			fmt.Printf("  %s seed %d: deliveries %.2f → %.2f · monsters %.3f → %.3f\n", mode, s, ee[0], ll[0], ee[1], ll[1])
		}
		fmt.Printf("%-8s average: deliveries per ant per 10k ticks %.2f → %.2f · monsters killed %.3f → %.3f\n", mode, e[0], l[0], e[1], l[1])
	}
	os.Chdir(wd)
}
