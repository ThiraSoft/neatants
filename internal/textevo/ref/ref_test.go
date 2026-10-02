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
	// A zero output gives every token logit 0: uniform, log2(50) bits.
	if b := Bits(d, o, 1, 7); math.Abs(b-math.Log2(50)) > 1e-9 {
		t.Fatalf("uniform: %g bits", b)
	}
	// Pointing at a token's own row makes it the most likely one.
	copy(o, d.Row(7))
	if Bits(d, o, 3, 7) >= math.Log2(50) {
		t.Fatal("aligned output not better than uniform")
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
