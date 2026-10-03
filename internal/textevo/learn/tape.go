// Package learn computes exact gradients of the textevo loss with respect to
// the weights of an evolved network, by backpropagation through time on the
// CPU, and applies them to a genome. It is the reference of Lamarckian
// learning: the forward pass is the one of neat.Flat bit for bit, so the
// scores it records are the ones ref.Rows gives.
//
// The state banks of a network (see model.Net) are treated as constants: the
// gradient does not flow through the bank inputs into the gates that moved
// them, so the gate outputs get no gradient. With no bank the gradient is
// exact.
package learn

import (
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// Tape runs a network over a window and records what the backward pass needs.
// A Tape is reused from window to window to avoid allocations, and it is not
// safe for concurrent use.
type Tape struct {
	f      *neat.Flat
	n      int // nodes
	nw     int // edges
	np     int // plastic links
	pred   int
	banks  int
	dim    int
	ticks  int
	warm   int
	plastq []int32 // plastic entry of each edge, -1 when its weight is fixed

	vals  []float32 // ticks*n node values
	sums  []float32 // ticks*4n sums per (node, gate) slot
	cells []float32 // ticks*n memory cells
	wp    []float32 // ticks*np weights of the plastic links, before the tick's learning
	rows  []float32 // scored ticks * pred, o = 2v-1
	zeros []float32

	wcur  []float32
	in    []float32
	gates []float32

	// Backward state.
	dv, nxt, dc []float64
	wt          []float64
	gw          []float64
	dW, dEta    []float64
}

// Run runs a fresh network f of g over the window of length ticks starting at
// ids[start], the way ref.Rows does, and records the tape. f must be
// g.BuildNetwork().Flat() and is not modified.
func (tp *Tape) Run(g *neat.Genome, f *neat.Flat, d *prep.Data, ids []int32, start, length, warm int) {
	s := model.ShapeOf(g, d.Dim)
	n := f.Nodes()
	tp.f, tp.n, tp.nw, tp.np = f, n, len(f.From), len(f.Plastic)
	tp.pred, tp.banks, tp.dim, tp.ticks, tp.warm = s.Predictions(), s.Banks, d.Dim, length, warm
	tp.vals = grow(tp.vals, length*n)
	tp.sums = grow(tp.sums, length*n*int(neat.NumGates))
	tp.cells = grow(tp.cells, length*n)
	tp.wp = grow(tp.wp, length*tp.np)
	tp.rows = grow(tp.rows, max(length-warm, 0)*tp.pred)
	tp.zeros = grow(tp.zeros, n)
	clear(tp.zeros)
	tp.wcur = grow(tp.wcur, tp.nw)
	copy(tp.wcur, f.Weight)
	tp.in = grow(tp.in, s.Inputs())
	clear(tp.in)
	tp.gates = grow(tp.gates, s.Banks)
	for b := range tp.gates {
		tp.gates[b] = 0.5
	}
	tp.plastq = growI(tp.plastq, tp.nw)
	for k := range tp.plastq {
		tp.plastq[k] = -1
	}
	for q, p := range f.Plastic {
		tp.plastq[p.K] = int32(q)
	}

	gn := int(neat.NumGates)
	for t := range length {
		v := tp.vals[t*n : (t+1)*n]
		prev := tp.zeros
		if t > 0 {
			prev = tp.vals[(t-1)*n : t*n]
		}
		row := d.Row(ids[start+t])
		copy(tp.in, row)
		model.StepState(tp.in[d.Dim:], row, tp.gates, tp.banks)
		v[0] = 1
		for i := f.InStart; i < f.InEnd; i++ {
			v[i] = tp.in[i-f.InStart]
		}
		for q, p := range f.Plastic {
			tp.wp[t*tp.np+q] = tp.wcur[p.K]
		}
		sm := tp.sums[t*n*gn : (t+1)*n*gn]
		for _, ni := range f.Order {
			base := int(ni) * gn
			if f.Kind[ni] != uint8(neat.Memory) {
				x := tp.sum(v, prev, base)
				sm[base] = x
				v[ni] = sigmoid32(x)
				continue
			}
			x := tp.sum(v, prev, base+int(neat.GateIn))
			si := tp.sum(v, prev, base+int(neat.GateInput))
			sf := tp.sum(v, prev, base+int(neat.GateForget))
			so := tp.sum(v, prev, base+int(neat.GateOutput))
			sm[base+int(neat.GateIn)], sm[base+int(neat.GateInput)] = x, si
			sm[base+int(neat.GateForget)], sm[base+int(neat.GateOutput)] = sf, so
			ig := sigmoid01_32(si + 2)
			fg := sigmoid01_32(sf + 2)
			og := sigmoid01_32(so + 2)
			var cp float32
			if t > 0 {
				cp = tp.cells[(t-1)*n+int(ni)]
			}
			c := fg*cp + ig*fastTanh32(x)
			tp.cells[t*n+int(ni)] = c
			v[ni] = og * fastTanh32(c)
		}
		out := v[f.OutStart:f.OutEnd]
		copy(tp.gates, out[tp.pred:])
		if t >= warm {
			o := tp.rows[(t-warm)*tp.pred : (t-warm+1)*tp.pred]
			for j := range o {
				o[j] = 2*out[j] - 1
			}
		}
		for _, p := range f.Plastic {
			pre, post := v[p.From], v[p.To]
			w := tp.wcur[p.K] + p.Eta*post*(pre-post*tp.wcur[p.K])
			tp.wcur[p.K] = max(-8, min(8, w))
		}
	}
}

// sum is the weighted sum of a slot, in the order of Flat.Activate.
func (tp *Tape) sum(v, prev []float32, slot int) float32 {
	f, n := tp.f, int32(tp.n)
	var s float32
	for k := f.Off[slot]; k < f.Off[slot+1]; k++ {
		if src := f.From[k]; src < n {
			s += v[src] * tp.wcur[k]
		} else {
			s += prev[src-n] * tp.wcur[k]
		}
	}
	return s
}

// push sends the gradient ds of the sum of a slot to its sources and to the
// weights of its edges.
func (tp *Tape) push(v, prev []float32, dv, nxt []float64, slot int, ds float64) {
	if ds == 0 {
		return
	}
	f, n := tp.f, int32(tp.n)
	for k := f.Off[slot]; k < f.Off[slot+1]; k++ {
		src := f.From[k]
		var sv float64
		w := tp.wt[k]
		if src < n {
			sv = float64(v[src])
			dv[src] += ds * w
		} else {
			sv = float64(prev[src-n])
			nxt[src-n] += ds * w
		}
		if q := tp.plastq[k]; q >= 0 {
			tp.gw[q] += ds * sv
		} else {
			tp.dW[k] += ds * sv
		}
	}
}

// Rows returns the prediction rows o = 2v-1 of the scored ticks (t >= warm)
// of the last Run, each Predictions values long. They equal ref.Rows. The
// slices are reused by the next Run.
func (tp *Tape) Rows() [][]float32 {
	r := max(tp.ticks-tp.warm, 0)
	out := make([][]float32, r)
	for i := range out {
		out[i] = tp.rows[i*tp.pred : (i+1)*tp.pred]
	}
	return out
}

// Row returns the prediction row of the i-th scored tick of the last Run.
func (tp *Tape) Row(i int) []float32 { return tp.rows[i*tp.pred : (i+1)*tp.pred] }

// Scored is the number of scored ticks of the last Run.
func (tp *Tape) Scored() int { return max(tp.ticks-tp.warm, 0) }

// Back backpropagates through time the gradient dO[r] of the loss with
// respect to the row o of the r-th scored tick (Predictions values each, so
// len(dO) must be Scored()). It returns the gradient with respect to the
// starting weight of every edge of the Flat and to the plasticity rate of
// every plastic link. The slices belong to the Tape and are overwritten by the
// next Back.
func (tp *Tape) Back(dO [][]float64) (dW, dEta []float64) {
	f, n, gn := tp.f, tp.n, int(neat.NumGates)
	tp.dv, tp.nxt, tp.dc = growF(tp.dv, n), growF(tp.nxt, n), growF(tp.dc, n)
	clear(tp.dv)
	clear(tp.nxt)
	clear(tp.dc)
	tp.wt = growF(tp.wt, tp.nw)
	for k, w := range f.Weight {
		tp.wt[k] = float64(w)
	}
	tp.gw = growF(tp.gw, tp.np)
	clear(tp.gw)
	tp.dW, tp.dEta = growF(tp.dW, tp.nw), growF(tp.dEta, tp.np)
	clear(tp.dW)
	clear(tp.dEta)

	for t := tp.ticks - 1; t >= 0; t-- {
		v := tp.vals[t*n : (t+1)*n]
		prev := tp.zeros
		if t > 0 {
			prev = tp.vals[(t-1)*n : t*n]
		}
		sm := tp.sums[t*n*gn : (t+1)*n*gn]
		dv, nxt := tp.dv, tp.nxt
		if t >= tp.warm {
			for j, g := range dO[t-tp.warm] {
				dv[int(f.OutStart)+j] += 2 * g
			}
		}
		// The learning rule of this tick, which ran after the node values.
		for q, p := range f.Plastic {
			pre, post := v[p.From], v[p.To]
			w := tp.wp[t*tp.np+q]
			un := w + p.Eta*post*(pre-post*w)
			g := tp.gw[q]
			if un < -8 || un > 8 {
				tp.gw[q] = 0
				continue
			}
			pr, po, w64, eta := float64(pre), float64(post), float64(w), float64(p.Eta)
			tp.gw[q] = g * (1 - eta*po*po)
			dv[p.From] += g * eta * po
			dv[p.To] += g * eta * (pr - 2*po*w64)
			tp.dEta[q] += g * po * (pr - po*w64)
		}
		for q, p := range f.Plastic {
			tp.wt[p.K] = float64(tp.wp[t*tp.np+q])
		}
		for i := len(f.Order) - 1; i >= 0; i-- {
			ni := int(f.Order[i])
			base := ni * gn
			dvn := dv[ni]
			if f.Kind[ni] != uint8(neat.Memory) {
				tp.push(v, prev, dv, nxt, base, dvn*1.225*dTanh(2.45*float64(sm[base])))
				continue
			}
			sx := float64(sm[base+int(neat.GateIn)])
			ui := 0.5 * (float64(sm[base+int(neat.GateInput)]) + 2)
			uf := 0.5 * (float64(sm[base+int(neat.GateForget)]) + 2)
			uo := 0.5 * (float64(sm[base+int(neat.GateOutput)]) + 2)
			ig, fg, og := 0.5+0.5*tanh(ui), 0.5+0.5*tanh(uf), 0.5+0.5*tanh(uo)
			c := float64(tp.cells[t*n+ni])
			var cp float64
			if t > 0 {
				cp = float64(tp.cells[(t-1)*n+ni])
			}
			tc := tanh(c)
			dc := tp.dc[ni] + dvn*og*dTanh(c)
			tp.push(v, prev, dv, nxt, base+int(neat.GateIn), dc*ig*dTanh(sx))
			tp.push(v, prev, dv, nxt, base+int(neat.GateInput), dc*tanh(sx)*0.25*dTanh(ui))
			tp.push(v, prev, dv, nxt, base+int(neat.GateForget), dc*cp*0.25*dTanh(uf))
			tp.push(v, prev, dv, nxt, base+int(neat.GateOutput), dvn*tc*0.25*dTanh(uo))
			tp.dc[ni] = dc * fg
		}
		// What this tick sent to the previous one is what it receives.
		tp.dv, tp.nxt = nxt, dv
		clear(tp.nxt)
	}
	// What is carried to tick 0 is the gradient with respect to the starting
	// weight of a plastic link.
	for q, p := range f.Plastic {
		tp.dW[p.K] = tp.gw[q]
	}
	return tp.dW, tp.dEta
}

func grow(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	return s[:n]
}

func growI(s []int32, n int) []int32 {
	if cap(s) < n {
		return make([]int32, n)
	}
	return s[:n]
}

func growF(s []float64, n int) []float64 {
	if cap(s) < n {
		return make([]float64, n)
	}
	return s[:n]
}

// The activations below are those of neat, which keeps them private.

func sigmoid32(x float32) float32    { return 0.5 + 0.5*fastTanh32(2.45*x) }
func sigmoid01_32(x float32) float32 { return 0.5 + 0.5*fastTanh32(0.5*x) }

func fastTanh32(x float32) float32 {
	if x > 3 {
		return 1
	}
	if x < -3 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}

// tanh is fastTanh32 in float64.
func tanh(x float64) float64 {
	if x > 3 {
		return 1
	}
	if x < -3 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}

// dTanh is the derivative of tanh, zero where it saturates.
func dTanh(x float64) float64 {
	if x > 3 || x < -3 {
		return 0
	}
	x2 := x * x
	den := 27 + 9*x2
	return ((27+3*x2)*den - 18*x2*(27+x2)) / (den * den)
}
