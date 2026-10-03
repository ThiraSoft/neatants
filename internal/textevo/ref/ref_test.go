package ref

import (
	"math"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

func TestBitsIsLogSoftmax(t *testing.T) {
	d := model.Synthetic(50, 8, 100, 1)
	o := make([]float32, 8)
	// A zero output leaves the prior alone in the logits: the unigram.
	zero := Bits(d, o, 1, 7)
	if math.Abs(zero-model.UnigramBits(d, 7)) > 1e-9 {
		t.Fatalf("zero output: %g bits, unigram %g", zero, model.UnigramBits(d, 7))
	}
	// Pointing at a token's own row makes it more likely than the prior says.
	copy(o, d.Row(7))
	if Bits(d, o, 3, 7) >= zero {
		t.Fatal("aligned output not better than the unigram")
	}
}

func TestRowsAreFreshPerWindow(t *testing.T) {
	d := model.Synthetic(50, 8, 400, 2)
	g := model.Grown(3, 8, 300)
	a := Rows(g, d, d.Train, 10, 64, 16)
	b := Rows(g, d, d.Train, 10, 64, 16)
	if len(a) != 48 || len(a[0]) != 8 {
		t.Fatalf("%d rows of %d", len(a), len(a[0]))
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatal("same window, different rows: state leaked between calls")
			}
			// The Pade tanh can overshoot its bound by a float32 ulp or so.
			if a[i][j] < -1-1e-5 || a[i][j] > 1+1e-5 {
				t.Fatalf("output %g out of [-1,1]", a[i][j])
			}
		}
	}
}

func TestEvaluate(t *testing.T) {
	d := model.Synthetic(50, 8, 400, 4)
	gs := []*neat.Genome{model.Grown(1, 8, 100), model.Grown(2, 8, 100)}
	starts := []int{0, 100, 200}
	bpb := Evaluate(gs, d, d.Train, starts, 64, 16)
	for i, g := range gs {
		bits := 0.0
		for _, s := range starts {
			rows := Rows(g, d, d.Train, s, 64, 16)
			for r, o := range rows {
				bits += Bits(d, o, model.LogitScale(g, 8), d.Train[s+16+r+1])
			}
		}
		want := bits / float64(model.WindowBytes(d, d.Train, starts, 64, 16))
		if math.Abs(bpb[i]-want) > 1e-9 {
			t.Fatalf("genome %d: %g, want %g", i, bpb[i], want)
		}
	}
}

// With o = 0 every dot product is zero and the logits are the prior alone, so
// the network is exactly the add-one unigram of the train tokens.
func TestZeroOutputIsUnigram(t *testing.T) {
	d := model.Synthetic(200, 16, 3000, 5)
	model.Skew(d, 5)
	g := model.ZeroGenome(1, 16)
	starts := []int{0, 700, 1500, 2900 - 80}
	got := Evaluate([]*neat.Genome{g}, d, d.Train, starts, 80, 16)[0]
	bits := 0.0
	for _, s := range starts {
		for tk := 16; tk < 80; tk++ {
			bits += model.UnigramBits(d, d.Train[s+tk+1])
		}
	}
	want := bits / float64(model.WindowBytes(d, d.Train, starts, 80, 16))
	t.Logf("zero genome %.6f bpb, unigram %.6f, uniform %.6f", got, want, math.Log2(200)*64*4/float64(model.WindowBytes(d, d.Train, starts, 80, 16)))
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("zero genome %g bpb, unigram %g", got, want)
	}
}

// A network whose only link reads the first value of bank 0 must see a token
// of the past: a large first embedding value at tick 0 still moves its
// output ticks later, where the same network over tokens whose first value
// is zero stays at o = 0.
func TestStateSeesThePast(t *testing.T) {
	s := model.Shape{Dim: 8, Banks: 2}
	g := model.NewGenomeShape(1, s)
	// Node IDs as neat.NewGenome numbers them: bias 0, inputs 1..Inputs,
	// outputs after.
	g.Conns = []neat.ConnGene{{In: 1 + 8, Out: 1 + s.Inputs(), Weight: 5, Enabled: true, Innovation: 1}}
	d := model.Synthetic(4, 8, 40, 6)
	for j := range 4 {
		d.E[j*8] = 0
	}
	d.E[1*8] = 4 // token 1 is the one to remember
	for i := range d.Train {
		d.Train[i] = 2
	}
	quiet := Rows(g, d, d.Train, 0, 12, 0)
	d.Train[0] = 1
	loud := Rows(g, d, d.Train, 0, 12, 0)
	for tk := range 12 {
		if quiet[tk][0] != 0 {
			t.Fatalf("tick %d: o[0] = %g without the token, want 0", tk, quiet[tk][0])
		}
	}
	for _, tk := range []int{0, 3, 6} {
		if loud[tk][0] <= 0.01 {
			t.Fatalf("tick %d: o[0] = %g, the bank forgot the token of tick 0", tk, loud[tk][0])
		}
		if len(loud[tk]) != 8 {
			t.Fatalf("%d values a row, want the 8 predictions without the gates", len(loud[tk]))
		}
	}
	t.Logf("o[0] after the token: %v", []float32{loud[0][0], loud[3][0], loud[6][0], loud[11][0]})
}

// The tree scoring is a distribution over the tokens whatever the outputs,
// and with zero outputs it is the add-one unigram, as the softmax is.
func TestTreeBitsIsADistribution(t *testing.T) {
	d := model.Synthetic(300, 16, 3000, 12)
	model.Skew(d, 12)
	tr := model.BuildTree(d)
	zero := make([]float32, tr.Depth)
	o := make([]float32, tr.Depth)
	for k := range o {
		o[k] = float32(math.Sin(float64(3*k + 1)))
	}
	sum := 0.0
	for id := range int32(d.Vocab()) {
		if z := TreeBits(tr, zero, 7, id); math.Abs(z-model.UnigramBits(d, id)) > 1e-6 {
			t.Fatalf("token %d: zero output %g bits, unigram %g", id, z, model.UnigramBits(d, id))
		}
		sum += math.Exp2(-TreeBits(tr, o, 7, id))
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("probabilities sum to %g", sum)
	}
}

func TestEvaluateTree(t *testing.T) {
	d := model.Synthetic(50, 8, 400, 4)
	tr := model.BuildTree(d)
	s := model.Shape{Dim: 8, Banks: 1, Code: tr.Depth}
	gs := []*neat.Genome{model.GrownShape(1, s, 100), model.GrownShape(2, s, 100)}
	starts := []int{0, 100, 200}
	bpb := EvaluateTree(gs, d, tr, d.Train, starts, 64, 16)
	for i, g := range gs {
		bits := 0.0
		for _, st := range starts {
			rows := Rows(g, d, d.Train, st, 64, 16)
			if len(rows[0]) != tr.Depth {
				t.Fatalf("%d values a row, want %d", len(rows[0]), tr.Depth)
			}
			for r, o := range rows {
				bits += TreeBits(tr, o, model.TreeScale(g), d.Train[st+16+r+1])
			}
		}
		want := bits / float64(model.WindowBytes(d, d.Train, starts, 64, 16))
		if math.Abs(bpb[i]-want) > 1e-9 {
			t.Fatalf("genome %d: %g, want %g", i, bpb[i], want)
		}
	}
}
