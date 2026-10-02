package sim

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// BenchmarkTickPhases splits a mid-game tick into its phases, multi-core.
func BenchmarkTickPhases(b *testing.B) {
	if v := os.Getenv("NEATANTS_CHUNK"); v != "" {
		fmt.Sscan(v, &minBrainChunk)
	}
	w := perfWorld(b)
	var tGrid, tBrain, tAct, tRest time.Duration
	b.ResetTimer()
	for range b.N {
		w.Tick++
		t0 := time.Now()
		w.rebuildGrids()
		t1 := time.Now()
		if cap(w.outs) < len(w.Ants) {
			w.outs = make([][]float64, len(w.Ants), len(w.Ants)*2)
		}
		w.outs = w.outs[:len(w.Ants)]
		clear(w.outs)
		w.runBrains()
		t2 := time.Now()
		for i, a := range w.Ants {
			if a.Alive && w.outs[i] != nil {
				w.stepAnt(a, w.outs[i])
			}
		}
		t3 := time.Now()
		w.updateFireballs()
		w.updateMonsters()
		w.updateColonies()
		w.updateMarriages()
		w.rechargeElements()
		w.updateWaves()
		w.updateShop()
		w.updateWeird()
		w.updateWeather()
		w.updateFoodAndPhero()
		w.cleanup()
		w.Events = w.Events[:0]
		t4 := time.Now()
		tGrid += t1.Sub(t0)
		tBrain += t2.Sub(t1)
		tAct += t3.Sub(t2)
		tRest += t4.Sub(t3)
	}
	n := float64(b.N)
	us := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / n / 1000 }
	fmt.Printf("\ngrids %.1f µs · brains (parallel) %.1f µs · actions %.1f µs · rest %.1f µs\n", us(tGrid), us(tBrain), us(tAct), us(tRest))
}
