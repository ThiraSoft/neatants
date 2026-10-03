package gpu

import (
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/learn"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// championGenome is a genome of about the size of a real champion, 250 nodes
// and 1500 edges (the one of learn's benchmark): a grown genome with hidden
// nodes and random links added, some of them closing cycles, with memory nodes
// and plastic links.
func championGenome(seed int64, dim int) *neat.Genome {
	rng := rand.New(rand.NewSource(seed))
	g := model.Grown(seed, dim, 300)
	g.AddMemory()
	for i := 0; len(g.Nodes) < 250; i++ {
		g.Nodes = append(g.Nodes, neat.NodeGene{ID: 1000000 + i, Type: neat.Hidden})
	}
	first := 1 + g.NumInputs
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

// benchLearn runs generations of 200 champion-sized genomes over 4 windows
// of 128 ticks at dimension 32. mode: "gpu" EvaluateLearn, "cpu" EvaluateGrad
// then learn.FromRows, "grad" EvaluateGrad alone, "nomask" EvaluateLearn
// without the backward kernels.
func benchLearn(b *testing.B, mode string, pop int) {
	dev, err := vk.Open()
	if err != nil {
		b.Skip(err)
	}
	defer dev.Close()
	const dim, length, warm = 32, 128, 16
	d := model.Synthetic(12000, dim, 300000, 1)
	gs := make([]*neat.Genome, pop)
	for i := range gs {
		gs[i] = championGenome(int64(i+1), dim)
	}
	f := gs[0].BuildNetwork().Flat()
	b.Logf("genome 0: %d nodes, %d edges", f.Nodes(), len(f.From))
	e, err := New(dev, d)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	e.skipBack = mode == "nobwd"
	rng := rand.New(rand.NewSource(1))
	bytesOf := func(starts []int) int { return model.WindowBytes(d, d.Train, starts, length, warm) }
	_ = bytesOf
	var tot, pack, gpu, cpu time.Duration
	run := func() {
		starts := model.DrawStarts(rng, len(d.Train), 4, length)
		t0 := time.Now()
		switch mode {
		case "gpu", "nobwd":
			if _, _, _, _, err := e.EvaluateLearn(gs, starts, length, warm); err != nil {
				b.Fatal(err)
			}
		default:
			_, _, fit, dO, dS, err := e.EvaluateGrad(gs, starts, length, warm)
			if err != nil {
				b.Fatal(err)
			}
			if mode == "cpu" {
				t1 := time.Now()
				learn.FromRows(gs, fit, d, d.Train, starts, length, warm, dO, dS)
				cpu += time.Since(t1)
			}
		}
		tot += time.Since(t0)
		pack += e.Timing.Pack
		gpu += e.Timing.GPU
	}
	run() // buffers and pipelines
	tot, pack, gpu, cpu = 0, 0, 0, 0
	b.ResetTimer()
	for range b.N {
		run()
	}
	n := time.Duration(b.N)
	b.Logf("%s pop %d: per generation %v (pack %v, gpu %v, cpu backward %v)", mode, pop, tot/n, pack/n, gpu/n, cpu/n)
}

func BenchmarkLearnGPU(b *testing.B)   { benchLearn(b, "gpu", 200) }
func BenchmarkLearnCPU(b *testing.B)   { benchLearn(b, "cpu", 200) }
func BenchmarkLearnGrad(b *testing.B)  { benchLearn(b, "grad", 200) }
func BenchmarkLearnNoBwd(b *testing.B) { benchLearn(b, "nobwd", 200) }
