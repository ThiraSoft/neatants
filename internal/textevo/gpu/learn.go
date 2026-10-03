package gpu

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/learn"
	"github.com/ThiraSoft/neatants/neat"
)

//go:embed netrun_tape.spv
var netrunTapeSPV []byte

//go:embed netrun_tape_big.spv
var netrunTapeBigSPV []byte

//go:embed back.spv
var backSPV []byte

//go:embed reduce.spv
var reduceSPV []byte

// MaxTapeBytes caps the tape of a generation, which holds for every pair of
// genome and window the values, sums, cells and plastic weights of every
// tick (a tick takes 2n+B+4M+P floats for n nodes, B back copies, M memory
// nodes and P plastic links, about 2 KB at 250 nodes). A generation that
// needs more makes EvaluateLearn fail with ErrTape.
var MaxTapeBytes = 1 << 30

// ErrTape is wrapped by the error of EvaluateLearn when the tape of the
// generation would exceed MaxTapeBytes; the caller can learn on the CPU
// (EvaluateGrad and learn.FromRows) or lower the population, the windows or
// the length.
var ErrTape = errors.New("gpu: the tape of the generation is too big")

// learnState is what EvaluateLearn adds to an Evaluator, built on first use.
// The per-pair tape and gradient slices live on the card; only the gradient
// of each genome, summed over its windows, is read back.
type learnState struct {
	netPipe    [2]*vk.Pipeline
	reducePipe *vk.Pipeline
	// back holds the pipelines of the backward kernel, one for each set of
	// capacities (see backSpec), with their sets.
	back             map[[4]uint32]*backVariant
	tape, grad, out  *vk.Buffer
	outrd            *vk.Buffer
	capTape, capGrad int
	capOut           int
	netSet           [2][2]*vk.Set // [tokens][variant]
	reduceSet        *vk.Set
}

func (l *learnState) dropSets() {
	for i := range l.netSet {
		for c := range l.netSet[i] {
			if l.netSet[i][c] != nil {
				l.netSet[i][c].Close()
				l.netSet[i][c] = nil
			}
		}
	}
	for _, v := range l.back {
		if v.set != nil {
			v.set.Close()
			v.set = nil
		}
	}
	if l.reduceSet != nil {
		l.reduceSet.Close()
		l.reduceSet = nil
	}
}

func (l *learnState) close() {
	l.dropSets()
	for _, p := range []*vk.Pipeline{l.netPipe[0], l.netPipe[1], l.reducePipe} {
		if p != nil {
			p.Close()
		}
	}
	for _, v := range l.back {
		v.pipe.Close()
	}
	for _, b := range []*vk.Buffer{l.tape, l.grad, l.out, l.outrd} {
		if b != nil {
			b.Close()
		}
	}
	*l = learnState{}
}

// backVariant is the backward kernel for one set of capacities.
type backVariant struct {
	pipe *vk.Pipeline
	set  *vk.Set
}

// tier rounds x up to the next of 1, 1.5, 2, 3, 4, ... times min, so that a
// generation whose genomes grow a little does not need a new pipeline.
func tier(x, min int) int {
	t := min
	for t < x {
		if t%3 == 0 || t&(t-1) != 0 {
			t = t / 3 * 4
		} else {
			t = t / 2 * 3
		}
		if t%2 == 1 {
			t++
		}
	}
	return t
}

// backSpec is the capacities back.comp is made for, in the order of its
// specialization constants: values and back copies, memory cells, plastic
// links, and the room for the products of a piece, all rounded up from the
// largest genome of recs.
func backSpec(recs [][]uint32) [4]uint32 {
	vals, mem, pl, prod := 1, 1, 1, 1
	for _, rec := range recs {
		n, nOrder, nLevels, nPlastic := int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])
		levelOff := 8 + n
		offOff := levelOff + 2*(nLevels+1) + nOrder
		edges := int(rec[offOff+4*nOrder])
		ext := offOff + 4*nOrder + 1 + 2*edges + 4*nPlastic + recordPad
		vals, mem, pl = max(vals, n+int(rec[ext+1])), max(mem, int(rec[ext])), max(pl, nPlastic)
		for l := range nLevels {
			prod = max(prod, int(rec[levelOff+2*l+3]-rec[levelOff+2*l+1]))
		}
	}
	prod = max(prod, 2*min(pl, plasticPiece))
	return [4]uint32{uint32(tier(vals, 128)), uint32(tier(mem, 16)), uint32(tier(pl, 16)), uint32(tier(prod, 64))}
}

// backPush is the push constant block of back.comp.
type backPush struct {
	Pairs, Windows, Len, Warm, Dim, GoffOff, ListOff, TapeOff, Total uint32
	Scale                                                            float32
}

// reducePush is the one of reduce.comp.
type reducePush struct{ Genomes, Windows, TapeOff, Total uint32 }

// learnPlan is the table the kernels of EvaluateLearn read after the records
// of the upload, and the sizes of what they work on. Per pair (genome-major,
// as the pairs of the upload): the first float of its tape and of its
// gradient slice, then per genome: where its summed gradient goes and how
// many floats it has (edges, then plastic links).
type learnPlan struct {
	words                   int // of the table
	tape, grad, out         int // floats
	edges, plastic          []int
	pairTape, pairGrad, off []uint32
}

// planLearn sizes the tape and the gradients of recs, made by PackLearn, over
// windows windows of length ticks.
func planLearn(recs [][]uint32, windows, length int) (learnPlan, error) {
	var pl learnPlan
	g := len(recs)
	pl.edges, pl.plastic = make([]int, g), make([]int, g)
	pl.pairTape = make([]uint32, g*windows)
	pl.pairGrad = make([]uint32, g*windows)
	pl.off = make([]uint32, g)
	tape := 0
	for i, rec := range recs {
		e, p, stride := learnShape(rec)
		pl.edges[i], pl.plastic[i] = e, p
		pl.off[i] = uint32(pl.out)
		for w := range windows {
			// The offsets are 32-bit float indices.
			if tape+stride*length > math.MaxUint32 || pl.grad+e+p > math.MaxUint32 {
				return pl, fmt.Errorf("%w: more than %d floats", ErrTape, uint32(math.MaxUint32))
			}
			pl.pairTape[i*windows+w] = uint32(tape)
			pl.pairGrad[i*windows+w] = uint32(pl.grad)
			tape += stride * length
			pl.grad += e + p
		}
		pl.out += e + p
	}
	pl.tape = tape
	if 4*tape > MaxTapeBytes {
		return pl, fmt.Errorf("%w: %d genomes x %d windows x %d ticks need %d KB, the limit is %d KB: lower -pop, -windows or -len, or raise gpu.MaxTapeBytes",
			ErrTape, g, windows, length, 4*tape>>10, MaxTapeBytes>>10)
	}
	pl.words = 2*g*windows + 2*g
	return pl, nil
}

// fill writes the table into dst, which holds pl.words words.
func (pl *learnPlan) fill(dst []uint32) {
	n := copy(dst, pl.pairTape)
	n += copy(dst[n:], pl.pairGrad)
	n += copy(dst[n:], pl.off)
	for i, e := range pl.edges {
		dst[n+i] = uint32(e + pl.plastic[i])
	}
}

// learnReady builds the pipelines, makes the buffers hold the plan and binds
// the sets of kind, after e.grow and e.gradReady.
func (e *Evaluator) learnReady(kind int, pl learnPlan, spec [4]uint32) error {
	l := &e.lrn
	if l.reducePipe == nil {
		var err error
		for c, spv := range [2][]byte{netrunTapeSPV, netrunTapeBigSPV} {
			if l.netPipe[c], err = e.d.NewPipeline(spv, 5, 10*4); err != nil {
				return err
			}
		}
		if l.reducePipe, err = e.d.NewPipeline(reduceSPV, 3, 4*4); err != nil {
			return err
		}
	}
	for _, c := range []struct {
		b    **vk.Buffer
		cap  *int
		need int
		make func(int) (*vk.Buffer, error)
	}{
		{&l.tape, &l.capTape, 4 * pl.tape, func(n int) (*vk.Buffer, error) { return e.d.Local(n, vk.UsageStorage) }},
		{&l.grad, &l.capGrad, 4 * pl.grad, func(n int) (*vk.Buffer, error) { return e.d.Local(n, vk.UsageStorage) }},
		{&l.out, &l.capOut, 4 * pl.out, func(n int) (*vk.Buffer, error) { return e.d.Local(n, vk.UsageStorage|vk.UsageTransferSrc) }},
	} {
		if max(c.need, 4) <= *c.cap {
			continue
		}
		l.dropSets()
		if *c.b != nil {
			(*c.b).Close()
			*c.b, *c.cap = nil, 0
		}
		// A little room, not a doubling: the tape is large.
		n := max(c.need, 4)
		n += n / 8
		b, err := c.make(n)
		if err != nil {
			return err
		}
		*c.b, *c.cap = b, n
		if c.b == &l.out {
			l.dropSets()
			if l.outrd != nil {
				l.outrd.Close()
				l.outrd = nil
			}
			if l.outrd, err = e.d.Readback(n, vk.UsageTransferDst); err != nil {
				return err
			}
		}
	}
	for c := range 2 {
		var err error
		if l.netSet[kind][c] == nil {
			if l.netSet[kind][c], err = l.netPipe[c].NewSet([]*vk.Buffer{e.gen, e.tokens[kind], e.emb32, e.rows, l.tape}); err != nil {
				return err
			}
		}
	}
	v := l.back[spec]
	if v == nil {
		p, err := e.d.NewPipelineSpec(backSPV, 4, 10*4, spec[:])
		if err != nil {
			return err
		}
		if l.back == nil {
			l.back = map[[4]uint32]*backVariant{}
		}
		v = &backVariant{pipe: p}
		l.back[spec] = v
	}
	if v.set == nil {
		var err error
		if v.set, err = v.pipe.NewSet([]*vk.Buffer{e.gen, l.tape, e.grad.dO, l.grad}); err != nil {
			return err
		}
	}
	if l.reduceSet == nil {
		var err error
		if l.reduceSet, err = l.reducePipe.NewSet([]*vk.Buffer{e.gen, l.grad, l.out}); err != nil {
			return err
		}
	}
	return nil
}

// recordBack records the backward pass, the sum over the windows and the
// readback of the summed gradients.
func (e *Evaluator) recordBack(r *vk.Recorder, gen genLayout, pl learnPlan, spec [4]uint32, windows, length, warm int, scale float32) {
	l := &e.lrn
	total := len(pl.edges) * windows
	if !e.skipBack {
		// One dispatch for both classes of genomes: their pairs are listed
		// one after the other.
		push := backPush{uint32(total), uint32(windows), uint32(length), uint32(warm), uint32(e.data.Dim),
			uint32(gen.goffOff), uint32(gen.listOff), uint32(gen.tapeOff), uint32(total), scale}
		r.Dispatch(l.back[spec].set, uint32(total), unsafe.Pointer(&push))
		r.Barrier()
		rp := reducePush{uint32(len(pl.edges)), uint32(windows), uint32(gen.tapeOff), uint32(total)}
		r.Dispatch(l.reduceSet, uint32(len(pl.edges)), unsafe.Pointer(&rp))
		r.Barrier()
	}
	if !e.skipCopy {
		r.Copy(l.outrd, 0, l.out, 4*pl.out)
	}
}

// EvaluateLearn is EvaluateGrad with the backward pass through the networks on
// the card as well: it returns, for each genome that fits, the gradient of its
// bpb with respect to its weights, plasticity rates and logit trait, the same
// numbers as learn.FromRows gives from the rows of EvaluateGrad (so float32
// and in another order of summation, but deterministic), and reads back
// nothing else than those gradients. bpb, oversized and fit are those of
// EvaluateGrad; grads[i] is the gradient of gs[fit[i]]. Banks are constants
// for the gradient, see package learn.
//
// The forward pass records a tape of every tick of every pair of genome and
// window on the card, which the backward kernel reads: when it would exceed
// MaxTapeBytes the error wraps ErrTape. The softmax scoring only.
func (e *Evaluator) EvaluateLearn(gs []*neat.Genome, starts []int, length, warm int) (bpb []float64, oversized int, fit []int, grads []learn.Grad, err error) {
	if e.tree != nil {
		return nil, 0, nil, nil, errors.New("gpu: EvaluateLearn is for the softmax scoring, this evaluator scores a tree")
	}
	out := gradOut{learn: true}
	bpb, oversized, err = e.evaluate(gs, 0, e.data.Train, starts, length, warm, &out)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	return bpb, oversized, out.fit, out.grads, nil
}

// readLearn turns the readback of the summed gradients into grads: the edges
// back to the order of the Flat of each genome, in float64, and the trait
// from the sum of the dS of its rows.
func (e *Evaluator) readLearn(gs []*neat.Genome, fit []int, slots []packed, pl learnPlan, dS []float32, per int, bytes float64) []learn.Grad {
	outs := unsafe.Slice((*float32)(unsafe.Pointer(&e.lrn.outrd.Bytes()[0])), pl.out)
	grads := make([]learn.Grad, len(fit))
	parallel(len(fit), func(i int) {
		mv := slots[fit[i]].moved
		ne, np := pl.edges[i], pl.plastic[i]
		o := outs[pl.off[i] : int(pl.off[i])+ne+np]
		gr := learn.Grad{W: make([]float64, len(mv)), Eta: make([]float64, np)}
		for k, m := range mv {
			if m >= 0 {
				gr.W[k] = float64(o[m])
			}
		}
		for q := range np {
			gr.Eta[q] = float64(o[ne+q])
		}
		if len(gs[fit[i]].Traits) > 0 {
			s := 0.0
			for _, x := range dS[i*per : (i+1)*per] {
				s += float64(x)
			}
			gr.Trait = s / bytes * 19 / math.Sqrt(float64(e.data.Dim))
		}
		grads[i] = gr
	})
	return grads
}
