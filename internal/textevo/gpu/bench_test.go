package gpu

import (
	"encoding/json"
	"math/rand"
	"os"
	"sync"
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
	benchPop(b, 1000, make, mode)
}

// benchPop is benchGen for a population of n genomes.
func benchPop(b *testing.B, n int, make func(i int, dim int) *neat.Genome, mode string) {
	benchPopOn(b, nil, n, make, mode)
}

// benchPopOn is benchPop on the data d, or on benchData's when d is nil.
func benchPopOn(b *testing.B, d *prep.Data, n int, make func(i int, dim int) *neat.Genome, mode string) {
	d0, err := vk.Open()
	if err != nil {
		b.Skip(err)
	}
	defer d0.Close()
	if d == nil {
		d = benchData(b)
	}
	gs := make_pop(n, d.Dim, make)
	e, err := New(d0, d)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	e.skipXent = mode == "net" || mode == "none"
	e.skipNet = mode == "xent" || mode == "none"
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
	g := time.Duration(b.N)
	b.Logf("per generation: pack %v, gpu %v, sum %v", pack/g, gpu/g, sum/g)
}

func make_pop(n, dim int, mk func(i, dim int) *neat.Genome) []*neat.Genome {
	gs := make([]*neat.Genome, n)
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
func BenchmarkNone(b *testing.B)           { benchGen(b, gen0, "none") }

func grownBig(i, dim int) *neat.Genome { return model.Grown(int64(i), dim, 150) }
func BenchmarkNetrunBig(b *testing.B)  { benchGen(b, grownBig, "net") }

// The saturation curve of the card: ms per generation against population.
func BenchmarkPop200(b *testing.B)  { benchPop(b, 200, grown, "") }
func BenchmarkPop500(b *testing.B)  { benchPop(b, 500, grown, "") }
func BenchmarkPop1000(b *testing.B) { benchPop(b, 1000, grown, "") }
func BenchmarkPop2000(b *testing.B) { benchPop(b, 2000, grown, "") }

// realChampion is the best genome of the first pop-200 run (runs/ is not in
// git): 776 nodes, about 2400 edges, 197 memory cells, 187 plastic links at
// its last generation. Without the file, bigGrown builds one of that size.
var realChampion = sync.OnceValue(func() *neat.Genome {
	raw, err := os.ReadFile("../../../runs/first-p200/champion.json")
	if err != nil {
		return nil
	}
	var c struct {
		Genome *neat.Genome `json:"genome"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Genome == nil {
		return nil
	}
	return c.Genome
})

// bigGrown grows a genome to the size of the champions of the first runs by
// mutating it with structural rates raised far above the defaults.
func bigGrown(seed int64, dim int) *neat.Genome {
	add, mem, rm, hebb := neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate, neat.HebbRate
	defer func() { neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate, neat.HebbRate = add, mem, rm, hebb }()
	neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate, neat.HebbRate = 0.4, 0.2, 0, 1
	g := model.NewGenome(int(seed), dim)
	for len(g.Nodes) < 1+2*dim+500 {
		g.Mutate()
	}
	return g
}

// realistic is a population at the size real runs reach: mutated copies of
// the real champion, or of one big grown genome when the file is missing.
func realistic(i, dim int) *neat.Genome {
	base := realChampion()
	if base == nil || base.NumInputs != dim {
		base = bigGenome(dim)
	}
	g := base.Copy()
	g.ID = i + 1
	for range 3 {
		g.Mutate()
	}
	return g
}

var bigGenome = func() func(dim int) *neat.Genome {
	var once sync.Once
	var g *neat.Genome
	return func(dim int) *neat.Genome {
		once.Do(func() { g = bigGrown(1, dim) })
		return g
	}
}()

// Realistic genomes at the two population sizes of the first runs, the
// whole generation and each kernel alone.
func BenchmarkRealPop200(b *testing.B)   { benchPop(b, 200, realistic, "") }
func BenchmarkRealPop1000(b *testing.B)  { benchPop(b, 1000, realistic, "") }
func BenchmarkRealNet200(b *testing.B)   { benchPop(b, 200, realistic, "net") }
func BenchmarkRealNet1000(b *testing.B)  { benchPop(b, 1000, realistic, "net") }
func BenchmarkRealXent1000(b *testing.B) { benchPop(b, 1000, realistic, "xent") }
func BenchmarkRealNone1000(b *testing.B) { benchPop(b, 1000, realistic, "none") }
func BenchmarkBigGrownPop1000(b *testing.B) {
	benchPop(b, 1000, func(i, dim int) *neat.Genome { g := bigGenome(dim).Copy(); g.ID = i + 1; g.Mutate(); return g }, "")
}

// TestRealisticSizes reports what the realistic benchmarks run.
func TestRealisticSizes(t *testing.T) {
	for name, g := range map[string]*neat.Genome{"champion": realChampion(), "bigGrown": bigGrown(1, 128)} {
		if g == nil {
			continue
		}
		f := g.BuildNetwork().Flat()
		mem := 0
		for _, k := range f.Kind {
			if k == uint8(neat.Memory) {
				mem++
			}
		}
		t.Logf("%s: %d nodes, %d edges, %d memory, %d plastic, %d levels, fits %v", name, f.Nodes(), len(f.From), mem, len(f.Plastic), len(f.LevelStart)-1, Fits(f))
	}
}

// benchDense runs a first generation of dense genomes (every output reads all
// the inputs) on the prepared file of dimension dim, mutated a few times as
// the children of the first generations are.
func benchDense(b *testing.B, file string, dim int, mode string) {
	benchDenseState(b, file, dim, 0, mode)
}

// benchDenseState is benchDense for networks with banks state banks; the
// dense start reads dim of the dim*(1+banks) inputs, as neattext -links dim
// -state banks does.
func benchDenseState(b *testing.B, file string, dim, banks int, mode string) {
	d, err := prep.Load("../../../data/" + file)
	if err != nil {
		b.Logf("no data/%s, synthetic data of the same size", file)
		d = model.Synthetic(12000, dim, 300000, 1)
	}
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	neat.MinimalLinks = dim
	benchPopOn(b, d, 200, func(i, dim int) *neat.Genome {
		g := model.NewGenomeShape(i+1, model.Shape{Dim: dim, Banks: banks})
		for range 3 {
			g.Mutate()
		}
		return g
	}, mode)
}

func BenchmarkDensePop200D128(b *testing.B)      { benchDense(b, "prep.bin", 128, "") }
func BenchmarkDensePop200D64(b *testing.B)       { benchDense(b, "prep64.bin", 64, "") }
func BenchmarkDensePop200D32(b *testing.B)       { benchDense(b, "prep32.bin", 32, "") }
func BenchmarkDenseNet200D128(b *testing.B)      { benchDense(b, "prep.bin", 128, "net") }
func BenchmarkDenseNet200D64(b *testing.B)       { benchDense(b, "prep64.bin", 64, "net") }
func BenchmarkDenseNet200D32(b *testing.B)       { benchDense(b, "prep32.bin", 32, "net") }
func BenchmarkDensePop200D32State4(b *testing.B) { benchDenseState(b, "prep32.bin", 32, 4, "") }
func BenchmarkDenseNet200D32State4(b *testing.B) { benchDenseState(b, "prep32.bin", 32, 4, "net") }

// benchGradOn times, for the population of realistic genomes at pop n, the
// xent kernel alone (Evaluate) and the fused kernel alone (EvaluateGrad
// without the readback), each as the GPU time of the submission.
func benchGradOn(b *testing.B, file string, dim, n int, grad bool) {
	d, err := prep.Load("../../../data/" + file)
	if err != nil {
		b.Skipf("no data/%s", file)
	}
	dev, err := vk.Open()
	if err != nil {
		b.Skip(err)
	}
	defer dev.Close()
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	neat.MinimalLinks = dim
	gs := make_pop(n, d.Dim, func(i, dim int) *neat.Genome {
		g := model.NewGenomeShape(i+1, model.Shape{Dim: dim})
		for range 3 {
			g.Mutate()
		}
		return g
	})
	e, err := New(dev, d)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	e.skipNet, e.skipCopy = true, os.Getenv("GRADCOPY") == ""
	rng := rand.New(rand.NewSource(1))
	// netrun is left out, so the rows are whatever the buffer holds: the
	// kernels' cost does not depend on them.
	starts := model.DrawStarts(rng, len(d.Train), 4, 128)
	run := func() {
		var err error
		if grad {
			_, _, _, _, _, err = e.EvaluateGrad(gs, starts, 128, 16)
		} else {
			_, _, err = e.Evaluate(gs, starts, 128, 16)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
	run()
	b.ResetTimer()
	var gpu time.Duration
	for range b.N {
		run()
		gpu += e.Timing.GPU
	}
	b.ReportMetric(float64(gpu.Milliseconds())/float64(b.N), "ms-gpu")
	b.ReportMetric(float64(n*4*112), "rows")
}

func BenchmarkXentD32Pop200(b *testing.B)       { benchGradOn(b, "prep32.bin", 32, 200, false) }
func BenchmarkXentGradD32Pop200(b *testing.B)   { benchGradOn(b, "prep32.bin", 32, 200, true) }
func BenchmarkXentD128Pop1000(b *testing.B)     { benchGradOn(b, "prep.bin", 128, 1000, false) }
func BenchmarkXentGradD128Pop1000(b *testing.B) { benchGradOn(b, "prep.bin", 128, 1000, true) }
