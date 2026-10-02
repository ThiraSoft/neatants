package sim

import (
	"math/rand"
	"os"
	"testing"
)

// perfWorld returns a fresh world advanced to mid-game, without touching the
// real save.
func perfWorld(b *testing.B) *World {
	LoadConfig("config.yml")
	wd, _ := os.Getwd()
	os.Chdir(b.TempDir())
	defer os.Chdir(wd)
	rand.Seed(11)
	w := NewWorld()
	for range 20000 {
		w.Update()
		w.Events = w.Events[:0]
	}
	return w
}

// BenchmarkMidGameTick measures one full tick in a mid-game world.
func BenchmarkMidGameTick(b *testing.B) {
	w := perfWorld(b)
	ants, mons := 0, 0
	for _, a := range w.Ants {
		if a.Alive {
			ants++
		}
	}
	for _, m := range w.Monsters {
		if m.Alive {
			mons++
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		w.Update()
		w.Events = w.Events[:0]
	}
	b.StopTimer()
	b.ReportMetric(float64(ants), "ants")
	b.ReportMetric(float64(mons), "monsters")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(max(1, ants)), "ns/ant")
}
