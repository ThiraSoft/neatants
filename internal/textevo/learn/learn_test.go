package learn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
	"github.com/ThiraSoft/neatants/neat"
)

const (
	tDim    = 8
	tLen    = 40
	tWarm   = 8
	tVocab  = 40
	tTokens = 600
)

var tStarts = []int{0, 100, 250}

// features tells what a genome exercises: back edges, memory nodes and
// plastic links.
func features(g *neat.Genome) (back, mem, plastic int) {
	f := g.BuildNetwork().Flat()
	for _, s := range f.From {
		if int(s) >= f.Nodes() {
			back++
		}
	}
	for _, k := range f.Kind {
		if k == uint8(neat.Memory) {
			mem++
		}
	}
	return back, mem, len(f.Plastic)
}

// rich grows genomes until one has back edges, memory nodes and plastic
// links, and makes the plasticity strong enough to matter.
func rich(t *testing.T, s model.Shape, mutations int) *neat.Genome {
	t.Helper()
	for seed := int64(1); seed < 200; seed++ {
		g := model.GrownShape(seed, s, mutations)
		if b, m, p := features(g); b > 0 && m > 0 && p > 0 {
			for i := range g.Conns {
				if g.Conns[i].Hebb != 0 {
					g.Conns[i].Hebb = 0.2
				}
			}
			return g
		}
	}
	t.Fatal("no genome with back edges, memory nodes and plastic links")
	return nil
}

func TestRowsEqualRef(t *testing.T) {
	d := model.Synthetic(tVocab, tDim, tTokens, 1)
	for _, banks := range []int{0, 2} {
		s := model.Shape{Dim: tDim, Banks: banks}
		for seed := int64(1); seed <= 4; seed++ {
			g := model.GrownShape(seed, s, 200)
			want := ref.Rows(g, d, d.Train, 5, tLen, tWarm)
			var tp Tape
			tp.Run(g, g.BuildNetwork().Flat(), d, d.Train, 5, tLen, tWarm)
			got := tp.Rows()
			if len(got) != len(want) {
				t.Fatalf("%d rows, want %d", len(got), len(want))
			}
			for i := range want {
				for j := range want[i] {
					if got[i][j] != want[i][j] {
						t.Fatalf("banks %d seed %d row %d col %d: %v != %v", banks, seed, i, j, got[i][j], want[i][j])
					}
				}
			}
		}
	}
}

func TestBpbEqualsRef(t *testing.T) {
	d := model.Synthetic(tVocab, tDim, tTokens, 2)
	var gs []*neat.Genome
	for seed := int64(1); seed <= 5; seed++ {
		gs = append(gs, model.GrownShape(seed, model.Shape{Dim: tDim, Banks: int(seed % 3)}, 150))
	}
	got, _ := Gradient(gs, d, d.Train, tStarts, tLen, tWarm)
	want := ref.Evaluate(gs, d, d.Train, tStarts, tLen, tWarm)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("genome %d: %v != %v", i, got[i], want[i])
		}
	}
}

func TestSoftmaxGradFD(t *testing.T) {
	d := model.Synthetic(tVocab, tDim, 50, 3)
	rng := rand.New(rand.NewSource(1))
	o := make([]float32, tDim)
	for i := range o {
		o[i] = float32(rng.NormFloat64() * 0.5)
	}
	bits := func(o []float32, s float32) float64 { return ref.Bits(d, o, s, 7) }
	b, dO, dS := SoftmaxGrad(d, o, 2, 7)
	if b != bits(o, 2) {
		t.Fatalf("bits %v != %v", b, bits(o, 2))
	}
	const eps = 1e-2
	for k := range o {
		p, m := append([]float32(nil), o...), append([]float32(nil), o...)
		p[k] += eps
		m[k] -= eps
		fd := (bits(p, 2) - bits(m, 2)) / (2 * eps)
		if math.Abs(fd-dO[k]) > 1e-3*(1+math.Abs(fd)) {
			t.Errorf("dO[%d]: %v, fd %v", k, dO[k], fd)
		}
	}
	fd := (bits(o, 2+eps) - bits(o, 2-eps)) / (2 * eps)
	if math.Abs(fd-dS) > 1e-3*(1+math.Abs(fd)) {
		t.Errorf("dScale: %v, fd %v", dS, fd)
	}
}

// loss is the bpb of g on the test windows.
func loss(g *neat.Genome, d *prep.Data) float64 {
	return ref.Evaluate([]*neat.Genome{g}, d, d.Train, tStarts, tLen, tWarm)[0]
}

// moved returns a copy of g with its weights moved by eps*dirW, its plasticity
// rates by eps*dirEta, and its trait by eps*dirT.
func moved(g *neat.Genome, dirW, dirEta []float64, dirT, eps float64) *neat.Genome {
	c := g.Copy()
	f := g.BuildNetwork().Flat()
	genes := g.EdgeGenes()
	for k, x := range dirW {
		c.Conns[genes[k]].Weight += eps * x
	}
	for q, x := range dirEta {
		c.Conns[genes[f.Plastic[q].K]].Hebb += eps * x
	}
	if len(c.Traits) > 0 {
		c.Traits[0] += eps * dirT
	}
	return c
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(a []float64) float64 { return math.Sqrt(dot(a, a)) }

func unit(a []float64) []float64 {
	n := norm(a)
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] / n
	}
	return out
}

func TestFiniteDifferences(t *testing.T) {
	d := model.Synthetic(tVocab, tDim, tTokens, 4)
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 3; trial++ {
		g := rich(t, model.Shape{Dim: tDim}, 150+100*trial)
		g.Traits[0] = 0.3
		b, m, p := features(g)
		if b == 0 || m == 0 || p == 0 {
			t.Fatalf("back %d memory %d plastic %d", b, m, p)
		}
		_, grads := Gradient([]*neat.Genome{g}, d, d.Train, tStarts, tLen, tWarm)
		gr := grads[0]
		zeroW, zeroE := make([]float64, len(gr.W)), make([]float64, len(gr.Eta))
		check := func(name string, dirW, dirE []float64, dirT, eps float64) {
			an := dot(gr.W, dirW) + dot(gr.Eta, dirE) + gr.Trait*dirT
			// A sharp loss can bend within the step, so the finite
			// difference gets two step sizes and keeps the closer.
			rel, fd := math.Inf(1), 0.0
			for _, e := range []float64{eps, eps / 10} {
				f := (loss(moved(g, dirW, dirE, dirT, e), d) - loss(moved(g, dirW, dirE, dirT, -e), d)) / (2 * e)
				if r := math.Abs(an-f) / (math.Abs(f) + 1e-3); r < rel {
					rel, fd = r, f
				}
			}
			t.Logf("trial %d (back %d mem %d plastic %d) %-9s analytic %+.6g fd %+.6g rel %.2g", trial, b, m, p, name, an, fd, rel)
			if rel > 0.03 {
				t.Errorf("%s: analytic %v, finite differences %v", name, an, fd)
			}
		}
		check("W grad", unit(gr.W), zeroE, 0, 1e-2)
		rd := make([]float64, len(gr.W))
		for i := range rd {
			rd[i] = rng.NormFloat64()
		}
		check("W random", unit(rd), zeroE, 0, 1e-2)
		check("Eta", zeroW, unit(gr.Eta), 0, 1e-3)
		check("Trait", zeroW, zeroE, 1, 1e-3)
	}
}

func TestApplyLowersLoss(t *testing.T) {
	d := model.Synthetic(tVocab, tDim, tTokens, 6)
	g := rich(t, model.Shape{Dim: tDim}, 200)
	g.Traits[0] = 0.3
	before := g.Copy()
	bpb, grads := Gradient([]*neat.Genome{g}, d, d.Train, tStarts, tLen, tWarm)
	c := Apply(g, grads[0], 0.01, 0.001, 0.005)
	if c == g {
		t.Fatal("Apply returned its argument")
	}
	for i := range g.Conns {
		if g.Conns[i] != before.Conns[i] {
			t.Fatal("Apply modified the genome it was given")
		}
	}
	if g.Traits[0] != before.Traits[0] {
		t.Fatal("Apply modified the trait")
	}
	after := loss(c, d)
	t.Logf("bpb %v -> %v", bpb[0], after)
	if !(after < bpb[0]) {
		t.Fatalf("loss went from %v to %v", bpb[0], after)
	}
	if c.ID != g.ID || c.Origin != g.Origin {
		t.Fatal("ID or Origin changed")
	}
}

func TestBanksAreConstants(t *testing.T) {
	// With banks the gradient is an approximation, but it must run and give
	// finite numbers.
	d := model.Synthetic(tVocab, tDim, tTokens, 7)
	g := model.GrownShape(3, model.Shape{Dim: tDim, Banks: 2}, 200)
	_, grads := Gradient([]*neat.Genome{g}, d, d.Train, tStarts, tLen, tWarm)
	for _, x := range grads[0].W {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			t.Fatal("non finite gradient")
		}
	}
}

// FromRows with the rows Gradient computes on the CPU gives Gradient's result.
func TestFromRowsMatchesGradient(t *testing.T) {
	d := model.Synthetic(60, 8, 600, 31)
	gs := []*neat.Genome{model.Grown(1, 8, 200), model.Grown(2, 8, 200), model.Grown(3, 8, 200)}
	starts := []int{0, 150, 300}
	const length, warm = 40, 8
	_, want := Gradient(gs, d, d.Train, starts, length, warm)
	fit := []int{0, 2}
	var dO, dS []float32
	for _, gi := range fit {
		g := gs[gi]
		scale := model.LogitScale(g, d.Dim)
		for _, s := range starts {
			for r, o := range ref.Rows(g, d, d.Train, s, length, warm) {
				_, do, ds := SoftmaxGrad(d, o, scale, d.Train[s+warm+r+1])
				for _, x := range do {
					dO = append(dO, float32(x))
				}
				dS = append(dS, float32(ds))
			}
		}
	}
	got := FromRows(gs, fit, d, d.Train, starts, length, warm, dO, dS)
	for i, gi := range fit {
		for k := range got[i].W {
			if math.Abs(got[i].W[k]-want[gi].W[k]) > 1e-4*(1+math.Abs(want[gi].W[k])) {
				t.Fatalf("genome %d edge %d: %g, want %g", gi, k, got[i].W[k], want[gi].W[k])
			}
		}
		if math.Abs(got[i].Trait-want[gi].Trait) > 1e-4*(1+math.Abs(want[gi].Trait)) {
			t.Fatalf("genome %d trait: %g, want %g", gi, got[i].Trait, want[gi].Trait)
		}
	}
}

// A gradient with a NaN leaves the genome as it was.
func TestApplyIgnoresNaN(t *testing.T) {
	g := model.Grown(1, 8, 100)
	gr := Grad{W: make([]float64, len(g.BuildNetwork().Flat().From))}
	gr.W[0] = math.NaN()
	c := Apply(g, gr, 0.1, 0.1, 0.1)
	for i := range c.Conns {
		if c.Conns[i].Weight != g.Conns[i].Weight {
			t.Fatal("a NaN gradient changed a weight")
		}
	}
}
