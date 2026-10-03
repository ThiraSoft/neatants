package learn

import (
	"math/rand"
	"runtime"
	"sync"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// BenchmarkBack runs Run and Back for 200 champion-sized genomes over 4
// windows of 128 ticks, with a fixed random dO: the part of learning that
// stays on the CPU. One iteration is one generation.
func BenchmarkBack(b *testing.B) {
	const dim, pop, length, warm = 32, 200, 128, 16
	d := model.Synthetic(200, dim, 2000, 1)
	starts := []int{0, 300, 600, 900}
	gs := make([]*neat.Genome, pop)
	fs := make([]*neat.Flat, pop)
	for i := range gs {
		gs[i] = champion(int64(i+1), dim)
		fs[i] = gs[i].BuildNetwork().Flat()
	}
	b.Logf("genome 0: %d nodes, %d edges", fs[0].Nodes(), len(fs[0].From))
	rng := rand.New(rand.NewSource(1))
	dO := make([][]float64, length-warm)
	for i := range dO {
		dO[i] = make([]float64, dim)
		for k := range dO[i] {
			dO[i][k] = rng.NormFloat64() * 0.01
		}
	}
	b.ResetTimer()
	for range b.N {
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i := range gs {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				var tp Tape
				for _, s := range starts {
					tp.Run(gs[i], fs[i], d, d.Train, s, length, warm)
					tp.Back(dO)
				}
			}()
		}
		wg.Wait()
	}
}

// champion returns a genome of about the size of a real champion, 250 nodes
// and 1500 edges: a grown genome with hidden nodes and random links added,
// some of them closing cycles, with memory nodes and plastic links.
func champion(seed int64, dim int) *neat.Genome {
	rng := rand.New(rand.NewSource(seed))
	g := model.Grown(seed, dim, 300)
	g.AddMemory()
	for i := 0; len(g.Nodes) < 250; i++ {
		g.Nodes = append(g.Nodes, neat.NodeGene{ID: 1000000 + i, Type: neat.Hidden})
	}
	first := 1 + g.NumInputs // outputs and hidden nodes can be read or written
	for len(g.Conns) < 1500 {
		out := first + rng.Intn(len(g.Nodes)-first)
		in := rng.Intn(len(g.Nodes))
		c := neat.ConnGene{In: g.Nodes[in].ID, Out: g.Nodes[out].ID, Weight: rng.NormFloat64() * 0.5,
			Enabled: true, Innovation: 100000 + len(g.Conns)}
		if g.Nodes[out].Type == neat.Memory {
			c.Gate = uint8(rng.Intn(int(neat.NumGates)))
		}
		g.Conns = append(g.Conns, c)
	}
	for i := 0; i < len(g.Conns); i += 97 {
		g.Conns[i].Hebb = 0.01
	}
	return g
}
