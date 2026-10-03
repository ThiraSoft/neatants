package gpu

import (
	_ "embed"
	"errors"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

//go:embed xent_grad.spv
var xentGradSPV []byte

//go:embed xent_grad_coop32.spv
var xentGradCoop32SPV []byte

//go:embed xent_grad_coop64.spv
var xentGradCoop64SPV []byte

//go:embed xent_grad_coop128.spv
var xentGradCoop128SPV []byte

// xentGradCoopSPV holds xent_grad_coop.comp built for each embedding width.
var xentGradCoopSPV = map[int][]byte{32: xentGradCoop32SPV, 64: xentGradCoop64SPV, 128: xentGradCoop128SPV}

// xentGradRows is how many rows one workgroup of xent_grad.comp covers, and
// xentGradCoopRows one of xent_grad_coop.comp.
const (
	xentGradRows     = 64
	xentGradCoopRows = 128
)

// newXentGradPipe builds the scalar fused kernel and, when coopDim is not
// zero, the matrix one for that width as its variant 1, as newXentPipe does.
func newXentGradPipe(d *vk.Device, coopDim int, wave uint32) (*vk.Pipeline, error) {
	p, err := d.NewPipeline(xentGradSPV, 8, 8*4)
	if err != nil || coopDim == 0 {
		return p, err
	}
	spv, ok := xentGradCoopSPV[coopDim]
	if !ok {
		p.Close()
		return nil, fmt.Errorf("gpu: xent_grad_coop is not built for dimension %d", coopDim)
	}
	if err := p.WideWave(1, spv, wave); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// gradState is what EvaluateGrad adds to an Evaluator, built on first use so
// that Evaluate pays nothing for it: the pipeline, the buffers the kernel
// writes (dO, dS, on the card) and the ones they are copied to for reading,
// and the sets that bind them for each token buffer.
type gradState struct {
	pipe         *vk.Pipeline
	dO, dS       *vk.Buffer
	dOrd, dSrd   *vk.Buffer
	capDO, capDS int
	set          [2]*vk.Set
}

func (g *gradState) dropSets() {
	for i, s := range g.set {
		if s != nil {
			s.Close()
			g.set[i] = nil
		}
	}
}

func (g *gradState) close() {
	g.dropSets()
	if g.pipe != nil {
		g.pipe.Close()
	}
	for _, b := range []*vk.Buffer{g.dO, g.dS, g.dOrd, g.dSrd} {
		if b != nil {
			b.Close()
		}
	}
	*g = gradState{}
}

// gradReady builds the pipeline, makes the buffers hold rows rows of dim floats
// and binds the sets of kind, after e.grow made the shared buffers.
func (e *Evaluator) gradReady(kind, rows int) error {
	g := &e.grad
	if g.pipe == nil {
		coopDim := 0
		if e.coop {
			coopDim = e.data.Dim
		}
		p, err := newXentGradPipe(e.d, coopDim, xentCoopWave)
		if err != nil {
			return err
		}
		g.pipe = p
	}
	dim := e.data.Dim
	if need := rows * dim * 4; need > g.capDO {
		g.dropSets()
		for _, b := range []**vk.Buffer{&g.dO, &g.dOrd} {
			if *b != nil {
				(*b).Close()
				*b = nil
			}
		}
		n := max(need, 2*g.capDO)
		g.capDO = 0
		var err error
		if g.dO, err = e.d.Local(n, vk.UsageStorage|vk.UsageTransferSrc); err != nil {
			return err
		}
		if g.dOrd, err = e.d.Readback(n, vk.UsageTransferDst); err != nil {
			return err
		}
		g.capDO = n
	}
	if need := rows * 4; need > g.capDS {
		g.dropSets()
		for _, b := range []**vk.Buffer{&g.dS, &g.dSrd} {
			if *b != nil {
				(*b).Close()
				*b = nil
			}
		}
		n := max(need, 2*g.capDS)
		g.capDS = 0
		var err error
		if g.dS, err = e.d.Local(n, vk.UsageStorage|vk.UsageTransferSrc); err != nil {
			return err
		}
		if g.dSrd, err = e.d.Readback(n, vk.UsageTransferDst); err != nil {
			return err
		}
		g.capDS = n
	}
	if g.set[kind] == nil {
		var err error
		g.set[kind], err = g.pipe.NewSet([]*vk.Buffer{e.gen, e.tokens[kind], e.emb16, e.rows, e.bits, e.prior, g.dO, g.dS})
		return err
	}
	return nil
}

// gradOut is what a call of evaluate with the gradient fills in.
type gradOut struct {
	fit    []int
	dO, dS []float32
}

// EvaluateGrad is Evaluate plus the gradient of the bits of every scored row
// with respect to what the network produced: the output vector o of the row
// and the logit scale s of its genome, both in bits. bpb, oversized and the
// reuse of records are those of Evaluate (bpb is +Inf for a genome too big
// for the kernel). The softmax scoring only: an Evaluator made by NewTree
// returns an error.
//
// Only the genomes that fit have rows. fit lists their indices in gs in
// increasing order, and row r of the results belongs to the genome fit[g]
// where, with D = the embedding dimension, W = len(starts) and
// T = length-warm scored ticks, rows are ordered by
//
//	r = (g*W + w)*T + t,   g the position in fit, w the window, t = tick-warm,
//
// so there are len(fit)*W*T rows. dO holds D floats for row r at dO[r*D:(r+1)*D],
// the derivative of the bits of row r with respect to the components of o
// (the output o = 2v-1 that Evaluate's kernel reads, not v). dS[r] is the
// derivative with respect to s, the model.LogitScale of the genome. These are
// the derivatives of the bits of each row, not of bpb: divide by the bytes of
// the windows (model.WindowBytes) and sum over the rows for the gradient of
// bpb. The slices are the caller's: a later call does not change them.
func (e *Evaluator) EvaluateGrad(gs []*neat.Genome, starts []int, length, warm int) (bpb []float64, oversized int, fit []int, dO, dS []float32, err error) {
	if e.tree != nil {
		return nil, 0, nil, nil, nil, errors.New("gpu: EvaluateGrad is for the softmax scoring, this evaluator scores a tree")
	}
	var out gradOut
	bpb, oversized, err = e.evaluate(gs, 0, e.data.Train, starts, length, warm, &out)
	if err != nil {
		return nil, 0, nil, nil, nil, err
	}
	return bpb, oversized, out.fit, out.dO, out.dS, nil
}

// recordGrad dispatches the fused kernel and copies its results to the
// readback buffers.
func (e *Evaluator) recordGrad(r *vk.Recorder, kind, rows int, push unsafe.Pointer) {
	if !e.skipXent {
		if e.coop {
			r.DispatchWide(e.grad.set[kind], 1, uint32((rows+xentGradCoopRows-1)/xentGradCoopRows), push)
		} else {
			r.Dispatch(e.grad.set[kind], uint32((rows+xentGradRows-1)/xentGradRows), push)
		}
		r.Barrier()
	}
	if !e.skipCopy {
		r.Copy(e.grad.dOrd, 0, e.grad.dO, rows*e.data.Dim*4)
		r.Copy(e.grad.dSrd, 0, e.grad.dS, rows*4)
	}
}
