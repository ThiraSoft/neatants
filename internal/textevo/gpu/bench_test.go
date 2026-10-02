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

func BenchmarkGeneration(b *testing.B) {
	d0, err := vk.Open()
	if err != nil {
		b.Skip(err)
	}
	defer d0.Close()
	d, err := prep.Load("../../../data/prep.bin")
	if err != nil {
		b.Logf("no data/prep.bin, synthetic data of the same size")
		d = model.Synthetic(12000, 128, 300000, 1)
	}
	gs := make([]*neat.Genome, 1000)
	for i := range gs {
		gs[i] = model.Grown(int64(i), d.Dim, 30)
	}
	e, err := New(d0, d)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
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
