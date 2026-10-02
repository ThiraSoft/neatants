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
// returns its output vector o = 2v-1 for each scored tick t >= warm.
func Rows(g *neat.Genome, d *prep.Data, ids []int32, start, length, warm int) [][]float32 {
	f := g.BuildNetwork().Flat()
	st := f.NewState()
	rows := make([][]float32, 0, length-warm)
	for t := range length {
		out := f.Activate(st, d.Row(ids[start+t]))
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

// Bits is -log2 of the softmax probability of target, in float64.
func Bits(d *prep.Data, o []float32, logitScale float32, target int32) float64 {
	m := math.Inf(-1)
	logits := make([]float64, d.Vocab())
	for j := range logits {
		var dot float32
		for k, e := range d.Row(int32(j)) {
			dot += e * o[k]
		}
		logits[j] = float64(logitScale * dot)
		m = math.Max(m, logits[j])
	}
	s := 0.0
	for _, x := range logits {
		s += math.Exp(x - m)
	}
	return (m + math.Log(s) - logits[target]) / math.Ln2
}

// Evaluate returns the bits per byte of each genome over the given windows,
// scoring the ticks from warm on. Genomes run in parallel, one per core.
func Evaluate(gs []*neat.Genome, d *prep.Data, ids []int32, starts []int, length, warm int) []float64 {
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
			scale := model.LogitScale(g, d.Dim)
			bits := 0.0
			for _, s := range starts {
				for r, o := range Rows(g, d, ids, s, length, warm) {
					bits += Bits(d, o, scale, ids[s+warm+r+1])
				}
			}
			out[i] = bits / bytes
		}()
	}
	wg.Wait()
	return out
}
