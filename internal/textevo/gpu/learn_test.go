package gpu

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/learn"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// featuresOf tells what a genome exercises: back edges, memory nodes and
// plastic links.
func featuresOf(g *neat.Genome) (back, mem, plastic int) {
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

// richGenome grows genomes until one has back edges, memory nodes and plastic
// links: it adds hidden nodes and random links (some of them closing cycles,
// some plastic, the plasticity strong enough to matter), since a mutated genome
// of the shapes with banks rarely has a cycle.
func richGenome(t *testing.T, s model.Shape, mutations int, from int64) *neat.Genome {
	t.Helper()
	for seed := from; seed < from+300; seed++ {
		g := model.GrownShape(seed, s, mutations)
		rng := rand.New(rand.NewSource(seed))
		for i := range 12 {
			g.Nodes = append(g.Nodes, neat.NodeGene{ID: 1000000 + i, Type: neat.Hidden})
		}
		first := 1 + g.NumInputs
		for i := range 40 {
			out := first + rng.Intn(len(g.Nodes)-first)
			in := rng.Intn(len(g.Nodes))
			c := neat.ConnGene{In: g.Nodes[in].ID, Out: g.Nodes[out].ID, Weight: rng.NormFloat64() * 0.5,
				Enabled: true, Innovation: 100000 + len(g.Conns)}
			if g.Nodes[out].Type == neat.Memory {
				c.Gate = uint8(rng.Intn(int(neat.NumGates)))
			}
			if i%4 == 0 {
				c.Hebb = 0.2
			}
			g.Conns = append(g.Conns, c)
		}
		for i := range g.Conns {
			if g.Conns[i].Hebb != 0 {
				g.Conns[i].Hebb = 0.2
			}
		}
		if b, m, p := featuresOf(g); b > 0 && m > 0 && p > 0 {
			return g
		}
	}
	t.Fatal("no genome with back edges, memory nodes and plastic links")
	return nil
}

// bulky is a genome of the big netrun variant: more than 256 memory nodes and
// than 256 plastic links (so several pieces of the Oja step), with cycles.
func bulky(s model.Shape, seed int64) *neat.Genome {
	g := model.GrownShape(seed, s, 100)
	for range 270 {
		g.AddMemory()
	}
	rng := rand.New(rand.NewSource(seed))
	first := 1 + g.NumInputs
	for i := range 700 {
		out := first + rng.Intn(len(g.Nodes)-first)
		in := rng.Intn(len(g.Nodes))
		c := neat.ConnGene{In: g.Nodes[in].ID, Out: g.Nodes[out].ID, Weight: rng.NormFloat64() * 0.3,
			Enabled: true, Innovation: 200000 + len(g.Conns)}
		if g.Nodes[out].Type == neat.Memory {
			c.Gate = uint8(rng.Intn(int(neat.NumGates)))
		}
		if i%2 == 0 {
			c.Hebb = 0.05
		}
		g.Conns = append(g.Conns, c)
	}
	return g
}

// relErr is the norm of a-b over the norm of b, with b's norm floored at tiny
// so that a zero gradient is not a division by zero.
func relErr(a []float32, b []float64) (rel, norm float64) {
	var num, den float64
	for i := range b {
		d := float64(a[i]) - b[i]
		num += d * d
		den += b[i] * b[i]
	}
	return math.Sqrt(num / math.Max(den, 1e-30)), math.Sqrt(den)
}

func f32(x []float64) []float32 {
	o := make([]float32, len(x))
	for i, v := range x {
		o[i] = float32(v)
	}
	return o
}

func TestEvaluateLearnMatchesCPU(t *testing.T) {
	for _, banks := range []int{0, 2} {
		for _, starts := range [][]int{{5, 800, 2000, 2500}, {1234}, {5, 900, 2111}} {
			t.Run(fmt.Sprintf("D32/banks%d/windows%d", banks, len(starts)), func(t *testing.T) { checkLearn(t, 32, banks, starts) })
		}
	}
	t.Run("D128/banks1", func(t *testing.T) { checkLearn(t, 128, 1, []int{5, 800, 2000}) })
}

func checkLearn(t *testing.T, D, banks int, starts []int) {
	dev := device(t)
	const V, length, warm = 333, 45, 7
	s := model.Shape{Dim: D, Banks: banks}
	d := model.Synthetic(V, D, 3000, int64(D+banks))
	for i := range d.E {
		d.E[i] = roundHalf(d.E[i])
	}
	var gs []*neat.Genome
	for i := range 3 {
		g := richGenome(t, s, 150+60*i, int64(1+1000*i))
		g.Traits[0] = 0.3
		gs = append(gs, g)
	}
	for i := range 3 {
		gs = append(gs, model.GrownShape(int64(40+i), s, 30*i))
	}
	// One genome too big for the kernel in the middle, and one without trait.
	big := model.NewGenomeShape(99, s)
	for range MaxNodes {
		big.AddMemory()
	}
	gs = append(gs[:2], append([]*neat.Genome{big}, gs[2:]...)...)
	gs[0].Traits = nil
	bk := bulky(s, 7)
	if f := bk.BuildNetwork().Flat(); netClass(f) != 1 || len(f.Plastic) <= plasticPiece {
		t.Fatalf("bulky genome: class %d, %d plastic links", netClass(f), len(f.Plastic))
	}
	gs = append(gs, bk)
	// The first generation: no hidden node, so no level sends a gradient to
	// another, and the output layer is cut in several pieces of 520 edges.
	gs = append(gs, model.NewGenomeShape(5, s))

	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	bpb, over, fit, grads, err := e.EvaluateLearn(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	if over != 1 || len(fit) != len(gs)-1 || len(grads) != len(fit) || !math.IsInf(bpb[2], 1) {
		t.Fatalf("over %d, fit %v, %d grads, bpb %v", over, fit, len(grads), bpb)
	}
	// The same rows through the CPU backward pass: only the backward kernel
	// differs.
	wantBpb, _, fit2, dO, dS, err := e.EvaluateGrad(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	for i := range fit {
		if fit[i] != fit2[i] {
			t.Fatalf("fit %v, %v", fit, fit2)
		}
	}
	for i := range gs {
		if bpb[i] != wantBpb[i] {
			t.Fatalf("genome %d: bpb %v, EvaluateGrad %v", i, bpb[i], wantBpb[i])
		}
	}
	want := learn.FromRows(gs, fit, d, d.Train, starts, length, warm, dO, dS)
	// The whole CPU, from its own forward pass and rows not rounded to halves,
	// which the scale amplifies: a loose check that the gradient is the one of
	// the loss.
	var fg []*neat.Genome
	for _, gi := range fit {
		fg = append(fg, gs[gi])
	}
	_, full := learn.Gradient(fg, d, d.Train, starts, length, warm)
	worst, worstFull := 0.0, 0.0
	for i := range fit {
		r, _ := relErr(f32(grads[i].W), full[i].W)
		worstFull = math.Max(worstFull, r)
	}
	t.Logf("worst relative error of W against learn.Gradient: %.3g", worstFull)
	if worstFull > 0.1 {
		t.Errorf("against learn.Gradient: %g", worstFull)
	}
	for i, gi := range fit {
		g, w := grads[i], want[i]
		if len(g.W) != len(w.W) || len(g.Eta) != len(w.Eta) {
			t.Fatalf("genome %d: %d weights %d rates, want %d and %d", gi, len(g.W), len(g.Eta), len(w.W), len(w.Eta))
		}
		rw, nw := relErr(f32(g.W), w.W)
		re, ne := relErr(f32(g.Eta), w.Eta)
		if len(w.Eta) == 0 {
			re = 0
		}
		rt := 0.0
		if len(gs[gi].Traits) > 0 {
			rt = math.Abs(g.Trait-w.Trait) / (math.Abs(w.Trait) + 1e-9)
		} else if g.Trait != 0 {
			t.Fatalf("genome %d has no trait and a gradient %g", gi, g.Trait)
		}
		b, m, p := featuresOf(gs[gi])
		t.Logf("genome %d (back %d mem %d plastic %d): |W| %.3g err %.2g, |eta| %.3g err %.2g, trait err %.2g", gi, b, m, p, nw, rw, ne, re, rt)
		worst = math.Max(worst, math.Max(rw, math.Max(re, rt)))
		if nw == 0 {
			t.Errorf("genome %d: a zero gradient", gi)
		}
		for _, x := range g.W {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("genome %d: non finite gradient", gi)
			}
		}
	}
	t.Logf("worst relative error %.3g", worst)
	if worst > 1e-3 {
		t.Fatalf("worst relative error %g", worst)
	}
}

func TestEvaluateLearnTapeLimit(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(100, 32, 1000, 3)
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	old := MaxTapeBytes
	MaxTapeBytes = 1 << 10
	defer func() { MaxTapeBytes = old }()
	_, _, _, _, err = e.EvaluateLearn([]*neat.Genome{model.Grown(1, 32, 50)}, []int{0}, 40, 8)
	if !errors.Is(err, ErrTape) {
		t.Fatalf("error %v", err)
	}
	t.Log(err)
}

func TestEvaluateLearnRefusesTree(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(100, 32, 1000, 3)
	e, err := NewTree(dev, d, model.BuildTree(d))
	if err != nil {
		t.Skip(err)
	}
	defer e.Close()
	if _, _, _, _, err := e.EvaluateLearn(nil, []int{0}, 40, 8); err == nil {
		t.Fatal("no error")
	}
}

// The gradient does not depend on the timing, on the records of the
// generation before, or on what ran in between: two calls give the very same
// numbers, with an Evaluate and an EvaluateGrad between them.
func TestEvaluateLearnRepeatable(t *testing.T) {
	dev := device(t)
	const length, warm = 40, 8
	s := model.Shape{Dim: 32, Banks: 1}
	d := model.Synthetic(200, 32, 2000, 9)
	var gs []*neat.Genome
	for i := range 5 {
		g := richGenome(t, s, 100, int64(1+500*i))
		gs = append(gs, g.Copy())
	}
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{0, 700, 1500}
	_, _, fit, a, err := e.EvaluateLearn(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := e.Evaluate(gs, starts, length, warm); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, err := e.EvaluateGrad(gs, starts, length, warm); err != nil {
		t.Fatal(err)
	}
	_, _, fit2, b, err := e.EvaluateLearn(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	if len(fit) != len(fit2) || len(a) != len(b) {
		t.Fatal("different sizes")
	}
	for i := range a {
		for k := range a[i].W {
			if a[i].W[k] != b[i].W[k] {
				t.Fatalf("genome %d edge %d: %g then %g", i, k, a[i].W[k], b[i].W[k])
			}
		}
		for q := range a[i].Eta {
			if a[i].Eta[q] != b[i].Eta[q] {
				t.Fatalf("genome %d rate %d: %g then %g", i, q, a[i].Eta[q], b[i].Eta[q])
			}
		}
	}
}
