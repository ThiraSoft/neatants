package sim

import (
	"math/rand"
	"testing"

	"github.com/ThiraSoft/neatants/neat"
)

// TestFlatThinksLikeTheCPU evaluates every Thought of a running world both
// ways for a few hundred ticks. Each network gets its flat twin at its first
// thought, as the GPU arena does.
func TestFlatThinksLikeTheCPU(t *testing.T) {
	LoadConfig("config.yml")
	rand.Seed(9)
	w := NewWorld()
	w.SerialBrains = true
	type twin struct {
		f *neat.Flat
		s *neat.FlatState
	}
	twins := map[*neat.Network]twin{}
	var in [max(AntInputs, MonInputs)]float32
	checked := 0
	for range 600 {
		w.Sense()
		for _, th := range w.Thoughts {
			tw, ok := twins[th.Net]
			if !ok {
				f := th.Net.Flat()
				tw = twin{f, f.NewState()}
				twins[th.Net] = tw
			}
			for i, x := range th.In {
				in[i] = float32(x)
			}
			got := tw.f.Activate(tw.s, in[:len(th.In)])
			think(th.Net, th.In, th.Out)
			for i := range th.Out {
				if float64(got[i]) != th.Out[i] {
					t.Fatalf("tick %d: flat output %d = %v, cpu %v", w.Tick, i, got[i], th.Out[i])
				}
			}
			checked++
		}
		w.Act()
	}
	if checked == 0 {
		t.Fatal("no thought was checked")
	}
}
