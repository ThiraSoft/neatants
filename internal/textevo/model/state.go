package model

import (
	"fmt"

	"github.com/ThiraSoft/neatants/neat"
)

// Decays of the state banks: half-lives of about 1, 5, 23 and 86 tokens. A
// network has at most len(Decays) banks, and bank b always has decay b, so
// that a network with fewer banks keeps the fastest ones.
var Decays = []float32{0.5, 0.875, 0.97, 0.992}

// MaxBanks is the largest number of state banks a network can have.
const MaxBanks = 4

// Shape is what a network reads and writes: the embedding of the current
// token followed by Banks state banks of Dim values, and its prediction
// followed by one gate a bank. The prediction is Dim values, a vector in
// embedding space, or Code values, one a level of a Tree, when Code is set.
type Shape struct{ Dim, Banks, Code int }

// Inputs is the number of input nodes of a genome of this shape.
func (s Shape) Inputs() int { return s.Dim * (1 + s.Banks) }

// Predictions is the number of prediction outputs of this shape.
func (s Shape) Predictions() int {
	if s.Code > 0 {
		return s.Code
	}
	return s.Dim
}

// Outputs is the number of output nodes of a genome of this shape.
func (s Shape) Outputs() int { return s.Predictions() + s.Banks }

// NewGenomeShape returns a minimal genome of shape s with one heritable
// trait, the logit scale. neat.MinimalLinks counts the state inputs like the
// embedding ones, so a dense start reads the banks too.
func NewGenomeShape(id int, s Shape) *neat.Genome {
	g := neat.NewGenome(id, s.Inputs(), s.Outputs())
	g.Traits = neat.RandomTraits(1)
	return g
}

// ShapeOf returns the shape of g for a dim-wide embedding. Every genome of a
// run has the shape its first generation was built with, so a genome that is
// not Dim*(1+B) inputs and P+B outputs for some B in [0, MaxBanks], with P
// either Dim or a code shorter than Dim, is a programming error, and it
// panics.
func ShapeOf(g *neat.Genome, dim int) Shape {
	b := -1
	if dim > 0 {
		b = g.NumInputs/dim - 1
	}
	s := Shape{Dim: dim, Banks: b}
	if p := g.NumOutputs - b; p >= 1 && p < dim {
		s.Code = p
	}
	if b < 0 || b > MaxBanks || g.NumInputs != s.Inputs() || g.NumOutputs != s.Outputs() {
		panic(fmt.Sprintf("textevo: a genome of %d inputs and %d outputs is not a network for embeddings of %d values with 0 to %d state banks",
			g.NumInputs, g.NumOutputs, dim, MaxBanks))
	}
	return s
}

// StepState moves the banks towards row; gates are the previous tick's gate
// outputs (sigmoids in [0,1]). A gate of 0.5 moves bank b by 1-Decays[b] of
// the way, a closed gate leaves it, an open one moves it twice as far.
func StepState(state, row, gates []float32, banks int) {
	dim := len(row)
	for b := range banks {
		r := (1 - Decays[b]) * 2 * gates[b]
		s := state[b*dim : (b+1)*dim]
		for k, x := range row {
			s[k] += r * (x - s[k])
		}
	}
}

// Net runs a genome one token at a time with its state banks, the way the
// evaluation runs it: before each tick the banks move towards the token's
// embedding with the gates of the tick before (0.5 before the first), the
// network reads the embedding then the banks, and its gate outputs are kept
// for the next tick.
type Net struct {
	f     *neat.Flat
	st    *neat.FlatState
	dim   int
	pred  int
	banks int
	in    []float32 // the embedding then the banks, which live here
	gates []float32
}

// NewNet returns a fresh network of g for a dim-wide embedding.
func NewNet(g *neat.Genome, dim int) *Net {
	s := ShapeOf(g, dim)
	f := g.BuildNetwork().Flat()
	n := &Net{f: f, st: f.NewState(), dim: dim, pred: s.Predictions(), banks: s.Banks, in: make([]float32, s.Inputs()), gates: make([]float32, s.Banks)}
	for b := range n.gates {
		n.gates[b] = 0.5
	}
	return n
}

// Step reads one embedding row and returns the prediction outputs v (not yet
// mapped to 2v-1). The slice is reused by the next call.
func (n *Net) Step(row []float32) []float32 {
	copy(n.in, row)
	StepState(n.in[n.dim:], row, n.gates, n.banks)
	out := n.f.Activate(n.st, n.in)
	copy(n.gates, out[n.pred:])
	return out[:n.pred]
}
