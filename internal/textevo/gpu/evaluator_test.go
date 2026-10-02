package gpu

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
	"github.com/ThiraSoft/neatants/neat"
)

func TestEvaluatorMatchesCPU(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 3000, 7)
	var gs []*neat.Genome
	for i := range 30 {
		gs = append(gs, model.Grown(int64(i), 32, 40*i))
	}
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{3, 900, 2000, 2500}
	got, over, err := e.Evaluate(gs, starts, 80, 16)
	if err != nil || over != 0 {
		t.Fatal(err, over)
	}
	want := ref.Evaluate(gs, d, d.Train, starts, 80, 16)
	for i := range gs {
		if math.Abs(got[i]-want[i]) > 2e-3*want[i] {
			t.Fatalf("genome %d: gpu %g cpu %g", i, got[i], want[i])
		}
	}
	// Twice in a row gives the same: no state survives between generations.
	again, _, _ := e.Evaluate(gs, starts, 80, 16)
	for i := range gs {
		if again[i] != got[i] {
			t.Fatalf("genome %d changed between two evaluations", i)
		}
	}
}

// The wide dimension makes xent's k-loop run over several 32-wide tiles.
func TestEvaluatorMatchesCPUWideDim(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 128, 3000, 11)
	var gs []*neat.Genome
	for i := range 8 {
		gs = append(gs, model.Grown(int64(i), 128, 40*i))
	}
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{3, 900, 2000, 2500}
	got, over, err := e.Evaluate(gs, starts, 80, 16)
	if err != nil || over != 0 {
		t.Fatal(err, over)
	}
	want := ref.Evaluate(gs, d, d.Train, starts, 80, 16)
	for i := range gs {
		if math.Abs(got[i]-want[i]) > 2e-3*want[i] {
			t.Fatalf("genome %d: gpu %g cpu %g", i, got[i], want[i])
		}
	}
}

func TestEvaluatorOversized(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 3000, 8)
	big := model.NewGenome(1, 32)
	for range MaxNodes { // one node per call: past MaxNodes in any case
		big.AddMemory()
	}
	small := model.Grown(2, 32, 50)
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got, over, err := e.Evaluate([]*neat.Genome{small, big, small}, []int{0, 100}, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	if over != 1 || !math.IsInf(got[1], 1) || got[0] != got[2] || math.IsInf(got[0], 0) {
		t.Fatalf("over %d, bpb %v", over, got)
	}
}

func TestValidateMatchesCPU(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 40000, 9)
	g := model.Grown(3, 32, 300)
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got, err := e.Validate(g)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.Evaluate([]*neat.Genome{g}, d, d.Val, model.ValStarts(len(d.Val)), model.ValLen, model.Warm)[0]
	if math.Abs(got-want) > 2e-3*want {
		t.Fatalf("gpu %g cpu %g", got, want)
	}
}

// An unchanged copy of a genome of the previous generation reuses its record,
// and the scores must be the ones a fresh evaluator gives.
func TestEvaluatorReusesChampionRecords(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 128, 3000, 13)
	var gs []*neat.Genome
	for i := range 12 {
		g := model.Grown(int64(i+1), 128, 20*i)
		g.ID = i + 1
		gs = append(gs, g)
	}
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{3, 900, 2000, 2500}
	if _, _, err := e.Evaluate(gs, starts, 80, 16); err != nil {
		t.Fatal(err)
	}
	// The next generation: copies of three genomes, one genome mutated, new ones.
	next := []*neat.Genome{model.Grown(100, 128, 30)}
	next[0].ID = 100
	for _, i := range []int{2, 5, 9} {
		c := gs[i].Copy()
		c.Origin, c.ID = gs[i].Lineage(), 200+i
		next = append(next, c)
	}
	m := gs[7].Copy()
	m.ID = 300
	m.Mutate()
	next = append(next, m)
	got, _, err := e.Evaluate(next, starts, 80, 16)
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	want, _, err := fresh.Evaluate(next, starts, 80, 16)
	if err != nil {
		t.Fatal(err)
	}
	for i := range next {
		if got[i] != want[i] {
			t.Fatalf("genome %d: with reuse %g, without %g", i, got[i], want[i])
		}
	}
}

// A copy that keeps its Origin but whose genes changed without Mutate (same
// node and connection counts) must not reuse the record of its origin.
func TestEvaluatorCacheIgnoresChangedCopy(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 3000, 17)
	g := model.Grown(5, 32, 60)
	g.ID = 1
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{3, 900, 2000}
	if _, _, err := e.Evaluate([]*neat.Genome{g}, starts, 80, 16); err != nil {
		t.Fatal(err)
	}
	c := g.Copy()
	c.Origin, c.ID = g.Lineage(), 2
	for i := range c.Conns {
		c.Conns[i].Weight = -c.Conns[i].Weight
	}
	got, _, err := e.Evaluate([]*neat.Genome{c}, starts, 80, 16)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.Evaluate([]*neat.Genome{c}, d, d.Train, starts, 80, 16)
	if math.Abs(got[0]-want[0]) > 2e-3*want[0] {
		t.Fatalf("changed copy: gpu %g cpu %g (stale record reused)", got[0], want[0])
	}
}

// A network with o = 0 is the unigram on the card too, through both xent
// kernels: D 32 always runs the scalar one, D 128 the matrix one when the
// device has it.
func TestEvaluatorZeroIsUnigram(t *testing.T) {
	for _, dim := range []int{32, 128} {
		dev := device(t)
		d := model.Synthetic(300, dim, 3000, 19)
		model.Skew(d, 19)
		e, err := New(dev, d)
		if err != nil {
			t.Fatal(err)
		}
		starts := []int{3, 900, 2000, 2500}
		got, _, err := e.Evaluate([]*neat.Genome{model.ZeroGenome(1, dim), model.ZeroGenome(2, dim)}, starts, 80, 16)
		e.Close()
		if err != nil {
			t.Fatal(err)
		}
		bits := 0.0
		for _, s := range starts {
			for tk := 16; tk < 80; tk++ {
				bits += model.UnigramBits(d, d.Train[s+tk+1])
			}
		}
		want := bits / float64(model.WindowBytes(d, d.Train, starts, 80, 16))
		t.Logf("D %d (coop %v): gpu %.6f, unigram %.6f", dim, e.coop, got[0], want)
		for i := range got {
			if math.Abs(got[i]-want) > 1e-3 {
				t.Fatalf("D %d genome %d: gpu %g bpb, unigram %g", dim, i, got[i], want)
			}
		}
	}
}

func TestEvaluatorStateMatchesCPU(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 3000, 23)
	var gs []*neat.Genome
	for i := range 30 {
		gs = append(gs, model.GrownShape(int64(i), model.Shape{Dim: 32, Banks: 2}, 40*i))
	}
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{3, 900, 2000, 2500}
	got, over, err := e.Evaluate(gs, starts, 80, 16)
	if err != nil || over != 0 {
		t.Fatal(err, over)
	}
	want := ref.Evaluate(gs, d, d.Train, starts, 80, 16)
	worst := 0.0
	for i := range gs {
		worst = math.Max(worst, math.Abs(got[i]-want[i])/want[i])
		if math.Abs(got[i]-want[i]) > 2e-3*want[i] {
			t.Fatalf("genome %d: gpu %g cpu %g", i, got[i], want[i])
		}
	}
	t.Logf("worst relative difference %g", worst)
}

// A genome whose inputs and outputs are not those of a network for the
// data's dimension cannot be scored: the evaluator says so instead of reading
// past the embedding row.
func TestEvaluatorRejectsWrongShape(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 3000, 25)
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	defer func() {
		if r := recover(); r == nil || !strings.Contains(fmt.Sprint(r), "state banks") {
			t.Fatalf("recovered %v, want the shape panic", r)
		}
	}()
	e.Evaluate([]*neat.Genome{model.NewGenome(1, 16)}, []int{0}, 64, 16)
}
