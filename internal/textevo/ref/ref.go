// Package ref is the CPU reference of the textevo evaluation. The GPU kernels
// are tested against it, so its semantics are the specification.
package ref

import (
	"math"
	"runtime"
	"sync"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// Rows runs a fresh network of g over the window starting at start and
// returns its output vector o = 2v-1 for each scored tick t >= warm. A genome
// with state banks (see model.Net) carries them through the window, and its
// gate outputs are not part of o.
func Rows(g *neat.Genome, d *prep.Data, ids []int32, start, length, warm int) [][]float32 {
	net := model.NewNet(g, d.Dim)
	rows := make([][]float32, 0, length-warm)
	for t := range length {
		out := net.Step(d.Row(ids[start+t]))
		if t < warm {
			continue
		}
		// out is reused by the next call, so it is copied while it is mapped.
		o := make([]float32, len(out))
		for j, v := range out {
			o[j] = 2*v - 1
		}
		rows = append(rows, o)
	}
	return rows
}

// Bits is -log2 of the softmax probability of target, in float64. The logit
// of token j is its unigram prior plus the scaled dot product, so that the
// network only has to learn what the token frequencies do not say.
func Bits(d *prep.Data, o []float32, logitScale float32, target int32) float64 {
	m := math.Inf(-1)
	prior := d.Prior()
	logits := make([]float64, d.Vocab())
	for j := range logits {
		var dot float32
		for k, e := range d.Row(int32(j)) {
			dot += e * o[k]
		}
		logits[j] = float64(prior[j]) + float64(logitScale*dot)
		m = math.Max(m, logits[j])
	}
	s := 0.0
	for _, x := range logits {
		s += math.Exp(x - m)
	}
	return (m + math.Log(s) - logits[target]) / math.Ln2
}

// TreeBits is -log2 of the probability the tree scoring gives target, in
// float64: at level k of its path the branch is decided by
// z = scale*o[k] + Bias[node], the second child having probability
// sigmoid(z).
func TreeBits(t *model.Tree, o []float32, scale float32, target int32) float64 {
	b := 0.0
	for k, w := range t.TokenPath(target) {
		if w == model.PathEnd {
			break
		}
		z := float64(scale*o[k]) + float64(t.Bias[w&^model.PathBit])
		if w&model.PathBit != 0 {
			z = -z
		}
		// -ln sigmoid(-z) = softplus(z), written so it never overflows.
		b += math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
	}
	return b / math.Ln2
}

// Evaluate returns the bits per byte of each genome over the given windows,
// scoring the ticks from warm on. Genomes run in parallel, one per core.
func Evaluate(gs []*neat.Genome, d *prep.Data, ids []int32, starts []int, length, warm int) []float64 {
	return evaluate(gs, d, ids, starts, length, warm, func(g *neat.Genome) func([]float32, int32) float64 {
		scale := model.LogitScale(g, d.Dim)
		return func(o []float32, target int32) float64 { return Bits(d, o, scale, target) }
	})
}

// EvaluateTree is Evaluate with the tree scoring, for genomes whose
// predictions are one output a level of t.
func EvaluateTree(gs []*neat.Genome, d *prep.Data, t *model.Tree, ids []int32, starts []int, length, warm int) []float64 {
	return evaluate(gs, d, ids, starts, length, warm, func(g *neat.Genome) func([]float32, int32) float64 {
		scale := model.TreeScale(g)
		return func(o []float32, target int32) float64 { return TreeBits(t, o, scale, target) }
	})
}

// evaluate runs the windows of every genome and adds up the bits that
// scorer(g) gives each scored row.
func evaluate(gs []*neat.Genome, d *prep.Data, ids []int32, starts []int, length, warm int,
	scorer func(g *neat.Genome) func(o []float32, target int32) float64) []float64 {
	out := make([]float64, len(gs))
	bytes := float64(model.WindowBytes(d, ids, starts, length, warm))
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	for i, g := range gs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			score := scorer(g)
			bits := 0.0
			for _, s := range starts {
				for r, o := range Rows(g, d, ids, s, length, warm) {
					bits += score(o, ids[s+warm+r+1])
				}
			}
			out[i] = bits / bytes
		}()
	}
	wg.Wait()
	return out
}
