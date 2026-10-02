package sim

import (
	"math"
	"runtime"
	"sync"

	"github.com/ThiraSoft/neatants/neat"
)

// A tick runs in three phases so that something else than this process's
// CPU can do the thinking (the headless runner batches the brains of many
// worlds on the GPU):
//
//	Sense   moves the clock, rebuilds the grids and fills the senses of every
//	        ant and monster that thinks this tick, listing them in Thoughts
//	think   evaluates each Thought: ThinkCPU here, or the GPU
//	Act     reads the outputs and runs the rest of the tick
//
// Update chains the three on the CPU, for the game and the tests.

// Thought is one brain evaluation: Sense fills In, the evaluator writes Out,
// Act reads Out. Both slices are backed by the ant or monster itself.
type Thought struct {
	Net *neat.Network
	In  []float64
	Out []float64
}

// Update runs a whole tick on the CPU and reports whether it is time to save.
func (w *World) Update() bool {
	w.Sense()
	w.ThinkCPU()
	return w.Act()
}

// Sense runs the first phase of a tick.
func (w *World) Sense() {
	w.Tick++
	w.Clock = math.Sin(float64(w.Tick) * clockFreq)
	w.Evolved++
	w.rebuildGrids()
	w.updateBushFood()

	// With think_every > 1 each ant thinks on its own turn (staggered by ID
	// so the load stays even) and keeps acting on its last decision between.
	every := max(1, Cfg.ThinkEvery)
	w.sensed = len(w.Ants)
	w.thinkers = w.thinkers[:0]
	w.Thoughts = w.Thoughts[:0]
	for _, a := range w.Ants {
		if a.Alive && (w.Tick+a.ID)%every == 0 {
			w.thinkers = append(w.thinkers, a)
			w.Thoughts = append(w.Thoughts, Thought{a.Net, a.Sense[:], a.lastOut[:]})
		}
	}
	w.parallel(len(w.thinkers), func(s, e int) {
		for _, a := range w.thinkers[s:e] {
			w.fillInputs(a, a.Sense[:])
		}
	})
	w.senseMonsters()
}

// ThinkCPU evaluates the Thoughts on this process's cores.
func (w *World) ThinkCPU() {
	w.parallel(len(w.Thoughts), func(s, e int) {
		for _, t := range w.Thoughts[s:e] {
			think(t.Net, t.In, t.Out)
		}
	})
}

// Act runs the rest of the tick and reports whether it is time to save.
func (w *World) Act() bool {
	// Only the ants that were there at Sense act: those hatched during this
	// phase start next tick. An ant killed by an earlier one is skipped.
	for _, a := range w.Ants[:w.sensed] {
		if a.Alive {
			w.stepAnt(a, a.lastOut[:])
		}
	}
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

	return w.Tick%Cfg.SaveEvery == 0
}

// think runs net on the float64 senses of the simulation and writes its
// outputs into out, which it returns. Networks compute in float32.
func think(net *neat.Network, in, out []float64) []float64 {
	var buf [max(AntInputs, MonInputs)]float32
	for i, x := range in {
		buf[i] = float32(x)
	}
	for i, x := range net.Activate(buf[:len(in)]) {
		out[i] = float64(x)
	}
	return out
}

// Workers live for the whole program and block on a channel between ticks:
// no goroutine (and no stack) is created or freed per tick.
type job struct {
	f    func(s, e int)
	s, e int
	wg   *sync.WaitGroup
}

var jobs = func() chan job {
	ch := make(chan job, 64)
	for range runtime.NumCPU() {
		go func() {
			for j := range ch {
				j.f(j.s, j.e)
				j.wg.Done()
			}
		}()
	}
	return ch
}()

// parallel splits [0, n) in chunks of at least minBrainChunk (waking a worker
// costs a few microseconds, tiny chunks are not worth it) and runs the first
// chunk on the calling goroutine itself. SerialBrains keeps it all here.
func (w *World) parallel(n int, f func(s, e int)) {
	if w.SerialBrains || n <= minBrainChunk {
		f(0, n)
		return
	}
	parts := min(runtime.NumCPU(), max(1, n/minBrainChunk))
	chunk := (n + parts - 1) / parts
	for s := chunk; s < n; s += chunk {
		w.wg.Add(1)
		jobs <- job{f, s, min(s+chunk, n), &w.wg}
	}
	f(0, min(chunk, n))
	w.wg.Wait()
}
