package gpu

import (
	"math"
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
