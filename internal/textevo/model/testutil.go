package model

import (
	"math"
	"math/rand"

	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// Synthetic builds a random corpus for tests: normal embeddings, random ids,
// and byte lengths cycling through 1..4. Train and Val are both tokens long.
func Synthetic(vocab, dim, tokens int, seed int64) *prep.Data {
	rng := rand.New(rand.NewSource(seed))
	d := &prep.Data{
		Dim:    dim,
		QwenID: make([]int32, vocab),
		Bytes:  make([]int32, vocab),
		E:      make([]float32, vocab*dim),
		Train:  make([]int32, tokens),
		Val:    make([]int32, tokens),
	}
	for i := range vocab {
		d.QwenID[i] = int32(i)
		d.Bytes[i] = int32(1 + i%4)
	}
	for i := range d.E {
		d.E[i] = float32(rng.NormFloat64())
	}
	for i := range tokens {
		d.Train[i] = int32(rng.Intn(vocab))
		d.Val[i] = int32(rng.Intn(vocab))
	}
	return d
}

// Grown returns a genome mutated mutations times, with two memory nodes and
// Hebbian plasticity on three links. The global rand source is unseeded, so
// the structure varies from run to run; tests must not depend on it.
func Grown(seed int64, dim, mutations int) *neat.Genome {
	g := NewGenome(int(seed), dim)
	for range mutations {
		g.Mutate()
	}
	g.AddMemory()
	g.AddMemory()
	var on []int
	for i := range g.Conns {
		if g.Conns[i].Enabled {
			on = append(on, i)
		}
	}
	if len(on) > 0 {
		for _, i := range []int{0, len(on) / 2, len(on) - 1} {
			g.Conns[on[i]].Hebb = 0.01
		}
	}
	return g
}

// ZeroGenome returns a genome whose outputs are exactly 0.5 whatever it reads,
// so that o = 2v-1 is exactly zero and only the prior is left in the logits.
func ZeroGenome(id, dim int) *neat.Genome {
	g := NewGenome(id, dim)
	for i := range g.Conns {
		g.Conns[i].Weight = 0
	}
	return g
}

// Skew redraws the train and validation ids of d from a Zipf law, so that the
// unigram prior of a test is far from uniform.
func Skew(d *prep.Data, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	z := rand.NewZipf(rng, 1.2, 1, uint64(d.Vocab()-1))
	for _, ids := range [][]int32{d.Train, d.Val} {
		for i := range ids {
			ids[i] = int32(z.Uint64())
		}
	}
}

// UnigramBits is -log2 of the softmax of the prior at target, in float64: what
// a network with o = 0 must score.
func UnigramBits(d *prep.Data, target int32) float64 {
	p := d.Prior()
	m := math.Inf(-1)
	for _, x := range p {
		m = math.Max(m, float64(x))
	}
	s := 0.0
	for _, x := range p {
		s += math.Exp(float64(x) - m)
	}
	return (m + math.Log(s) - float64(p[target])) / math.Ln2
}
