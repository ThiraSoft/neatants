package sim

import (
	"math/rand"
	"testing"
)

// BenchmarkTick measures one world tick with a fixed seed and population.
func BenchmarkTick(b *testing.B) {
	LoadConfig("config.yml")
	rand.Seed(1)
	w := NewWorld()
	for range 3000 {
		w.Update()
		w.Events = w.Events[:0]
	}
	b.ReportMetric(float64(len(w.Ants)), "ants")
	b.ResetTimer()
	for range b.N {
		w.Update()
		w.Events = w.Events[:0]
	}
}
