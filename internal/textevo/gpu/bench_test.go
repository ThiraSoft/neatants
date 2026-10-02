package gpu

import (
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// benchData loads the real prep file, or builds synthetic data of its size.
func benchData(b *testing.B) *prep.Data {
	d, err := prep.Load("../../../data/prep.bin")
	if err != nil {
		b.Logf("no data/prep.bin, synthetic data of the same size")
		d = model.Synthetic(12000, 128, 300000, 1)
	}
	return d
}

// benchGen runs generations of the population make builds and logs the split.
// mode picks the kernels: "" both, "net" netrun alone, "xent" xent alone.
func benchGen(b *testing.B, make func(i int, dim int) *neat.Genome, mode string) {
	d0, err := vk.Open()
	if err != nil {
		b.Skip(err)
	}
	defer d0.Close()
	d := benchData(b)
	gs := make_pop(d.Dim, make)
	e, err := New(d0, d)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	e.skipXent = mode == "net"
	e.skipNet = mode == "xent"
	rng := rand.New(rand.NewSource(1))
	b.ResetTimer()
	var pack, gpu, sum time.Duration
	for range b.N {
		if _, _, err := e.Evaluate(gs, model.DrawStarts(rng, len(d.Train), 4, 128), 128, 16); err != nil {
			b.Fatal(err)
		}
		pack += e.Timing.Pack
		gpu += e.Timing.GPU
		sum += e.Timing.Sum
	}
	n := time.Duration(b.N)
	b.Logf("per generation: pack %v, gpu %v, sum %v", pack/n, gpu/n, sum/n)
}

func make_pop(dim int, mk func(i, dim int) *neat.Genome) []*neat.Genome {
	gs := make([]*neat.Genome, 1000)
	for i := range gs {
		gs[i] = mk(i, dim)
	}
	return gs
}

func grown(i, dim int) *neat.Genome { return model.Grown(int64(i), dim, 30) }
func gen0(i, dim int) *neat.Genome  { return model.NewGenome(i, dim) }

func BenchmarkGeneration(b *testing.B)     { benchGen(b, grown, "") }
func BenchmarkGenerationGen0(b *testing.B) { benchGen(b, gen0, "") }
func BenchmarkNetrunGrown(b *testing.B)    { benchGen(b, grown, "net") }
func BenchmarkNetrunGen0(b *testing.B)     { benchGen(b, gen0, "net") }
func BenchmarkXentOnly(b *testing.B)       { benchGen(b, gen0, "xent") }
