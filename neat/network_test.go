package neat

import (
	"math"
	"math/rand"
	"testing"
)

// bigGenome grows a genome by mutation until it has hidden nodes, memory
// cells, plastic links and cycles, like the evolved ones.
func bigGenome(seed int64, inputs, outputs int) *Genome {
	rand.Seed(seed)
	g := NewGenomeWithHidden(1, inputs, outputs, 8)
	for range 3000 {
		g.Mutate()
	}
	// Mutate rarely closes a cycle on its own, and removal keeps memory cells
	// scarce, so add connections and memory cells directly. Random hidden to
	// hidden links soon close cycles.
	for range 5 {
		g.AddMemory()
	}
	for range 200 {
		g.addConnMutation()
	}
	return g
}

func randomInputs(r *rand.Rand, n int) ([]float32, []float64) {
	in32, in64 := make([]float32, n), make([]float64, n)
	for i := range in32 {
		x := r.Float64()*2 - 1
		in32[i], in64[i] = float32(x), float64(float32(x))
	}
	return in32, in64
}

// TestFloat32TracksFloat64 checks that the float32 network computes what the
// float64 one did: same structure, same back edge semantics, small drift.
func TestFloat32TracksFloat64(t *testing.T) {
	for seed := int64(1); seed <= 5; seed++ {
		g := bigGenome(seed, 67, 8)
		n32, n64 := g.BuildNetwork(), buildRef(g)
		r := rand.New(rand.NewSource(seed))
		for tick := range 50 {
			in32, in64 := randomInputs(r, 67)
			o32, o64 := n32.Activate(in32), n64.Activate(in64)
			for i := range o64 {
				if d := math.Abs(float64(o32[i]) - o64[i]); d > 2e-3 {
					t.Fatalf("seed %d tick %d output %d: float32 %v, float64 %v", seed, tick, i, o32[i], o64[i])
				}
			}
		}
	}
}

// TestGenomeHasCyclesAndMemory guards the test above: it is only meaningful
// if the genomes exercise back edges, memory cells and plasticity.
func TestGenomeHasCyclesAndMemory(t *testing.T) {
	g := bigGenome(1, 67, 8)
	n := g.BuildNetwork()
	back, mem := 0, 0
	for _, f := range n.from {
		if int(f) >= n.n {
			back++
		}
	}
	for _, k := range n.kind {
		if k == Memory {
			mem++
		}
	}
	if back == 0 || mem == 0 || len(n.plastic) == 0 {
		t.Fatalf("weak test genome: %d back edges, %d memory nodes, %d plastic links", back, mem, len(n.plastic))
	}
}
