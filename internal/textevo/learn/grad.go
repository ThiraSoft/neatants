package learn

import (
	"math"
	"runtime"
	"sync"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// Grad is the gradient of a loss with respect to what a genome can learn.
type Grad struct {
	W     []float64 // per edge of the Flat of the genome
	Eta   []float64 // per entry of Flat.Plastic
	Trait float64   // the logit scale trait, Traits[0]
}

// SoftmaxGrad is the CPU reference of the scoring: it returns the bits of
// target under the softmax of the logits prior + scale*dot(e_j, o), the
// gradient of the bits with respect to o, and with respect to scale. The bits
// are those of ref.Bits.
func SoftmaxGrad(d *prep.Data, o []float32, scale float32, target int32) (bits float64, dO []float64, dScale float64) {
	dO = make([]float64, len(o))
	bits, dScale = softmaxGrad(d, o, scale, target, make([]float64, 2*d.Vocab()), dO)
	return
}

// softmaxGrad is SoftmaxGrad with its buffers given: work holds twice the
// vocabulary, dO is overwritten.
func softmaxGrad(d *prep.Data, o []float32, scale float32, target int32, work, dO []float64) (bits, dScale float64) {
	v := d.Vocab()
	logits, dots := work[:v], work[v:2*v]
	prior := d.Prior()
	m := math.Inf(-1)
	for j := range logits {
		var dot float32
		for k, e := range d.Row(int32(j)) {
			dot += e * o[k]
		}
		dots[j] = float64(dot)
		logits[j] = float64(prior[j]) + float64(scale*dot)
		m = math.Max(m, logits[j])
	}
	s := 0.0
	for _, x := range logits {
		s += math.Exp(x - m)
	}
	bits = (m + math.Log(s) - logits[target]) / math.Ln2
	clear(dO)
	meanDot := 0.0
	for j, x := range logits {
		p := math.Exp(x-m) / s
		meanDot += p * dots[j]
		for k, e := range d.Row(int32(j)) {
			dO[k] += p * float64(e)
		}
	}
	for k, e := range d.Row(target) {
		dO[k] = float64(scale) * (dO[k] - float64(e)) / math.Ln2
	}
	dScale = (meanDot - dots[target]) / math.Ln2
	return
}

// Gradient returns the bits per byte of each genome over the windows (the
// same numbers as ref.Evaluate) and the gradient of that bpb with respect to
// the weights, plasticity rates and logit trait of the genome. Genomes run in
// parallel, one per core. Banks are constants for the gradient, see the
// package comment.
func Gradient(gs []*neat.Genome, d *prep.Data, ids []int32, starts []int, length, warm int) (bpb []float64, grads []Grad) {
	bpb = make([]float64, len(gs))
	grads = make([]Grad, len(gs))
	bytes := float64(model.WindowBytes(d, ids, starts, length, warm))
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	for i, g := range gs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			bpb[i], grads[i] = gradient(g, d, ids, starts, length, warm, bytes)
		}()
	}
	wg.Wait()
	return
}

func gradient(g *neat.Genome, d *prep.Data, ids []int32, starts []int, length, warm int, bytes float64) (float64, Grad) {
	f := g.BuildNetwork().Flat()
	scale := model.LogitScale(g, d.Dim)
	var tp Tape
	gr := Grad{W: make([]float64, len(f.From)), Eta: make([]float64, len(f.Plastic))}
	work := make([]float64, 2*d.Vocab())
	dO := make([][]float64, max(length-warm, 0))
	for i := range dO {
		dO[i] = make([]float64, d.Dim)
	}
	bits, dScale := 0.0, 0.0
	for _, s := range starts {
		tp.Run(g, f, d, ids, s, length, warm)
		dO = dO[:tp.Scored()]
		for r := range dO {
			dO[r] = dO[r][:tp.pred]
			b, ds := softmaxGrad(d, tp.Row(r), scale, ids[s+warm+r+1], work, dO[r])
			bits += b
			dScale += ds
			for k := range dO[r] {
				dO[r][k] /= bytes
			}
		}
		dW, dEta := tp.Back(dO)
		for k, x := range dW {
			gr.W[k] += x
		}
		for q, x := range dEta {
			gr.Eta[q] += x
		}
	}
	if len(g.Traits) > 0 {
		gr.Trait = dScale / bytes * 19 / math.Sqrt(float64(d.Dim))
	}
	return bits / bytes, gr
}

// Apply returns a copy of g after one step of normalized gradient descent: the
// weights move by lrW, the plasticity rates by lrEta, each group divided by
// the root mean square of its gradient, and the trait by lrTrait in the
// direction opposite to the sign of its gradient. Weights stay in [-8, 8],
// rates in [-1, 1] and the trait in [0, 1]. g itself is not modified, since
// evaluators cache records by genome pointer.
func Apply(g *neat.Genome, gr Grad, lrW, lrEta, lrTrait float64) *neat.Genome {
	c := g.Copy()
	if !gr.Finite() {
		// A NaN would go through the clamps below (max and min keep it)
		// and poison the genome for good.
		return c
	}
	f := g.BuildNetwork().Flat()
	genes := g.EdgeGenes()
	rw, re := rms(gr.W), rms(gr.Eta)
	for k, x := range gr.W {
		w := &c.Conns[genes[k]].Weight
		*w = max(-8, min(8, *w-lrW*x/(rw+1e-12)))
	}
	for q, x := range gr.Eta {
		h := &c.Conns[genes[f.Plastic[q].K]].Hebb
		*h = max(-1, min(1, *h-lrEta*x/(re+1e-12)))
	}
	if len(c.Traits) > 0 {
		switch {
		case gr.Trait > 0:
			c.Traits[0] -= lrTrait
		case gr.Trait < 0:
			c.Traits[0] += lrTrait
		}
		c.Traits[0] = max(0, min(1, c.Traits[0]))
	}
	return c
}

func rms(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	return math.Sqrt(s / float64(len(x)))
}

// FromRows is Gradient when the softmax already ran elsewhere, on the card:
// dO and dS are the gradients of the bits of each scored row with respect to
// o and to the logit scale, rows ordered by genome of fit, window, then tick
// (see gpu.Evaluator.EvaluateGrad), and only the backward pass through the
// networks runs here. grads[i] is the gradient of the bpb of gs[fit[i]].
func FromRows(gs []*neat.Genome, fit []int, d *prep.Data, ids []int32, starts []int, length, warm int, dO, dS []float32) []Grad {
	grads := make([]Grad, len(fit))
	bytes := float64(model.WindowBytes(d, ids, starts, length, warm))
	scored := length - warm
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup
	for i, gi := range fit {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			g := gs[gi]
			f := g.BuildNetwork().Flat()
			var tp Tape
			gr := Grad{W: make([]float64, len(f.From)), Eta: make([]float64, len(f.Plastic))}
			rows := make([][]float64, scored)
			for r := range rows {
				rows[r] = make([]float64, d.Dim)
			}
			dScale := 0.0
			for w, s := range starts {
				tp.Run(g, f, d, ids, s, length, warm)
				base := (i*len(starts) + w) * scored
				for r := range rows {
					src := dO[(base+r)*d.Dim : (base+r+1)*d.Dim]
					for k, x := range src {
						rows[r][k] = float64(x) / bytes
					}
					dScale += float64(dS[base+r])
				}
				dW, dEta := tp.Back(rows)
				for k, x := range dW {
					gr.W[k] += x
				}
				for q, x := range dEta {
					gr.Eta[q] += x
				}
			}
			if len(g.Traits) > 0 {
				gr.Trait = dScale / bytes * 19 / math.Sqrt(float64(d.Dim))
			}
			grads[i] = gr
		}()
	}
	wg.Wait()
	return grads
}

// Finite reports whether every value of the gradient is a number.
func (gr Grad) Finite() bool {
	for _, x := range gr.W {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	for _, x := range gr.Eta {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return !math.IsNaN(gr.Trait) && !math.IsInf(gr.Trait, 0)
}
