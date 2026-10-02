package sim

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// BenchmarkBrainParts times, per ant, the senses, the target vectors and the
// network, single-threaded.
func BenchmarkBrainParts(b *testing.B) {
	w := perfWorld(b)
	w.rebuildGrids()
	var in [AntInputs]float64
	var tIn, tTg, tNet time.Duration
	n := 0
	b.ResetTimer()
	for range b.N {
		for _, a := range w.Ants {
			if !a.Alive {
				continue
			}
			t0 := time.Now()
			w.fillInputs(a, in[:])
			t1 := time.Now()
			w.fillTargets(a, in[NumSenseDirs*SenseCh+AntStateIn:], math.Cos(a.Angle), math.Sin(a.Angle))
			t2 := time.Now()
			a.Net.Activate(in[:])
			t3 := time.Now()
			tIn += t1.Sub(t0) - t2.Sub(t1) // fillInputs includes one fillTargets call
			tTg += t2.Sub(t1)
			tNet += t3.Sub(t2)
			n++
		}
	}
	per := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / float64(n) }
	conns, nodes := 0, 0
	for _, a := range w.Ants {
		conns += len(a.Genome.Conns)
		nodes += len(a.Genome.Nodes)
	}
	fmt.Printf("\nper ant: sensors %.0f ns · target vectors %.0f ns · network %.0f ns (%.0f nodes, %.0f connections on average)\n",
		per(tIn), per(tTg), per(tNet), float64(nodes)/float64(len(w.Ants)), float64(conns)/float64(len(w.Ants)))
}
