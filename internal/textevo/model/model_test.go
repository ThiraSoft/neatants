package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/neatants/neat"
)

func TestDrawStartsLeaveRoomForTheTarget(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for range 1000 {
		for _, s := range DrawStarts(rng, 200, 4, 128) {
			if s < 0 || s+128 >= 200 { // reads ids[s .. s+128] inclusive
				t.Fatalf("start %d", s)
			}
		}
	}
	if s := DrawStarts(rng, 129, 2, 128); s[0] != 0 || s[1] != 0 {
		t.Fatalf("tight corpus: %v", s)
	}
}

func TestValStarts(t *testing.T) {
	s := ValStarts(100000)
	if len(s) != ValWindows || s[0] != 0 || s[len(s)-1]+ValLen >= 100000 {
		t.Fatalf("%v", s)
	}
}

func TestScale(t *testing.T) {
	g := NewGenome(1, 8)
	g.Traits[0] = 0
	if Scale(g) != 1 {
		t.Fatal(Scale(g))
	}
	g.Traits[0] = 1
	if Scale(g) != 20 {
		t.Fatal(Scale(g))
	}
}

// A dense start (neat.MinimalLinks = dim) must give every output all the
// inputs and the bias, so the -links flag of neattext reaches the genome.
func TestNewGenomeHonoursMinimalLinks(t *testing.T) {
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	for _, k := range []int{1, 5, 16} {
		neat.MinimalLinks = k
		g := NewGenome(1, 16)
		reads := map[int]int{}
		for _, c := range g.Conns {
			if c.In != 0 {
				reads[c.Out]++
			}
		}
		if len(reads) != 16 || len(g.Conns) != 16*(k+1) {
			t.Fatalf("links %d: %d outputs read inputs, %d connections", k, len(reads), len(g.Conns))
		}
		for out, n := range reads {
			if n != k {
				t.Fatalf("links %d: output %d reads %d inputs", k, out, n)
			}
		}
	}
}

func TestStepState(t *testing.T) {
	const dim = 4
	state := make([]float32, 2*dim)
	row := []float32{1, 1, 1, 1}
	StepState(state, row, []float32{0.5, 0.5}, 2)
	if state[0] != 1-Decays[0] || state[dim] != 1-Decays[1] {
		t.Fatalf("default gate: %v", state)
	}
	StepState(state, row, []float32{0, 0}, 2)
	if state[0] != 1-Decays[0] {
		t.Fatal("a closed gate must not write")
	}
	for range 200 {
		StepState(state, row, []float32{0.5, 0.5}, 2)
	}
	if math.Abs(float64(state[0]-1)) > 1e-3 {
		t.Fatalf("bank 0 should converge to the row: %v", state[0])
	}
}

func TestShape(t *testing.T) {
	s := Shape{Dim: 8, Banks: 3}
	g := NewGenomeShape(1, s)
	if g.NumInputs != 32 || g.NumOutputs != 11 || len(g.Traits) != 1 {
		t.Fatalf("%d inputs, %d outputs, %d traits", g.NumInputs, g.NumOutputs, len(g.Traits))
	}
	if got := ShapeOf(g, 8); got != s {
		t.Fatalf("ShapeOf %+v", got)
	}
	if got := ShapeOf(NewGenome(1, 8), 8); got != (Shape{8, 0}) {
		t.Fatalf("plain genome: %+v", got)
	}
	bad := []struct {
		g   *neat.Genome
		dim int
	}{
		{neat.NewGenome(1, 8, 9), 8},        // a gate output without its bank
		{neat.NewGenome(1, 20, 8), 8},       // inputs not a whole number of banks
		{NewGenomeShape(1, Shape{8, 1}), 4}, // read at another dimension
		{neat.NewGenome(1, 48, 13), 8},      // more banks than decays
	}
	for _, b := range bad {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%d inputs, %d outputs at dim %d did not panic", b.g.NumInputs, b.g.NumOutputs, b.dim)
				}
			}()
			ShapeOf(b.g, b.dim)
		}()
	}
}

// Net must give what a hand-written loop over Flat gives: the banks move
// with the token, are read after the embedding, and the gates of a tick come
// from the outputs after the D predictions of the tick before.
func TestNetFollowsStepState(t *testing.T) {
	s := Shape{Dim: 8, Banks: 2}
	d := Synthetic(20, 8, 100, 3)
	g := GrownShape(4, s, 200)
	net := NewNet(g, 8)
	f := g.BuildNetwork().Flat()
	st := f.NewState()
	state := make([]float32, 16)
	gates := []float32{0.5, 0.5}
	for tk := range 40 {
		row := d.Row(d.Train[tk])
		StepState(state, row, gates, 2)
		out := f.Activate(st, append(append([]float32(nil), row...), state...))
		copy(gates, out[8:])
		got := net.Step(row)
		if len(got) != 8 {
			t.Fatalf("%d outputs, want the 8 predictions", len(got))
		}
		for j := range got {
			if got[j] != out[j] {
				t.Fatalf("tick %d output %d: %g, want %g", tk, j, got[j], out[j])
			}
		}
	}
}
