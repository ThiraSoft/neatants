package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// encodePrompt maps a prompt to active ids, or fails naming the first token
// the corpus never uses.
func encodePrompt(enc prep.Encoder, d *prep.Data, text string) ([]int32, error) {
	var ids []int32
	for _, q := range enc.Encode(text, false, false) {
		// QwenID is sorted, so the active id is found by bisection.
		i := sort.Search(len(d.QwenID), func(i int) bool { return d.QwenID[i] >= q })
		if i == len(d.QwenID) || d.QwenID[i] != q {
			return nil, fmt.Errorf("prompt token %q (Qwen id %d) never appears in the corpus", enc.Decode([]int32{q}, false), q)
		}
		ids = append(ids, int32(i))
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("the prompt encodes to no token")
	}
	return ids, nil
}

// logits returns the scores of every active token for the output vector o,
// computed as the evaluation does: the unigram prior plus scale times the
// dot with the embedding.
func logits(d *prep.Data, o []float32, scale float32) []float64 {
	prior := d.Prior()
	out := make([]float64, d.Vocab())
	for j := range out {
		var dot float32
		for k, e := range d.Row(int32(j)) {
			dot += e * o[k]
		}
		out[j] = float64(prior[j]) + float64(scale*dot)
	}
	return out
}

// draw picks an index from the softmax of x/temp.
func draw(x []float64, temp float64, rng *rand.Rand) int32 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, v)
	}
	p := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		p[i] = math.Exp((v - m) / temp)
		sum += p[i]
	}
	r := rng.Float64() * sum
	for i, v := range p {
		if r -= v; r < 0 {
			return int32(i)
		}
	}
	return int32(len(p) - 1)
}

// sample runs the champion on the CPU: it reads the prompt, then draws each
// next token from its softmax at temperature temp.
func sample(g *neat.Genome, d *prep.Data, prompt []int32, n int, temp float64, rng *rand.Rand) []int32 {
	// model.Net carries the state banks of the champion, if it has any, the
	// way the evaluation ran it.
	net := model.NewNet(g, d.Dim)
	scale := model.LogitScale(g, d.Dim)
	o := make([]float32, d.Dim)
	read := func(id int32) {
		for j, v := range net.Step(d.Row(id)) {
			o[j] = 2*v - 1
		}
	}
	// The predictions made while reading the prompt are ignored: only the
	// state they leave behind matters.
	for _, id := range prompt {
		read(id)
	}
	out := make([]int32, 0, n)
	for range n {
		id := draw(logits(d, o, scale), temp, rng)
		out = append(out, id)
		read(id)
	}
	return out
}
