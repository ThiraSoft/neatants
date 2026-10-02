package sim

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// TestPhaseCost times each part of a tick, replaying Update step by step.
func TestPhaseCost(t *testing.T) {
	LoadConfig("config.yml")
	rand.Seed(1)
	w := NewWorld()
	for range 3000 {
		w.Update()
		w.Events = w.Events[:0]
	}
	cost := map[string]time.Duration{}
	step := func(name string, f func()) {
		t0 := time.Now()
		f()
		cost[name] += time.Since(t0)
	}
	const n = 3000
	ants := 0
	for range n {
		w.Tick++
		step("grids", w.rebuildGrids)
		outs := make([][]float64, len(w.Ants))
		step("sensors+networks (sequential)", func() {
			var in [AntInputs]float64
			for i, a := range w.Ants {
				if a.Alive {
					w.fillInputs(a, in[:])
					a.Sense = in
					o := think(a.Net, in[:], make([]float64, AntOutputs))
					outs[i] = append([]float64(nil), o...)
				}
			}
		})
		step("ant actions", func() {
			for i, a := range w.Ants {
				if a.Alive && outs[i] != nil {
					w.stepAnt(a, outs[i])
				}
			}
		})
		step("fireballs", w.updateFireballs)
		step("monsters", w.updateMonsters)
		step("colonies", w.updateColonies)
		step("waves", w.updateWaves)
		step("weather", w.updateWeather)
		step("food+pheromones", w.updateFoodAndPhero)
		step("cleanup", w.cleanup)
		w.Events = w.Events[:0]
		ants += len(w.Ants)
	}
	type kv struct {
		k string
		v time.Duration
	}
	var list []kv
	var total time.Duration
	for k, v := range cost {
		list = append(list, kv{k, v})
		total += v
	}
	sort.Slice(list, func(i, j int) bool { return list[i].v > list[j].v })
	fmt.Printf("%d ants, %d monsters on average\n", ants/n, len(w.Monsters))
	for _, e := range list {
		fmt.Printf("  %-32s %6.1f µs  %4.1f%%\n", e.k, float64(e.v.Nanoseconds())/1000/n, 100*float64(e.v)/float64(total))
	}
	fmt.Printf("  %-32s %6.1f µs\n", "sequential total", float64(total.Nanoseconds())/1000/n)
}
