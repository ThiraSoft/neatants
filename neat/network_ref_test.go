package neat

type refPlasticLink struct {
	k, from, to int32
	eta         float64
}

// refNetwork is the compiled, flat form of a genome. Incoming connections are
// stored contiguously per (node, gate) so activation walks linear memory.
type refNetwork struct {
	values []float64
	cell   []float64 // LSTM cell state, persists between ticks
	kind   []NodeType
	order  []int32
	off    []int32 // off[n*NumGates+g] .. off[n*NumGates+g+1] index into from/weight
	from   []int32
	weight []float64
	out    []float64
	// plastic lists the links whose weight learns during the life (Hebb != 0).
	plastic []refPlasticLink

	inStart, inEnd, outStart, outEnd int
}

func buildRef(g *Genome) *refNetwork {
	idMap := make(map[int]int, len(g.Nodes))
	for i, node := range g.Nodes {
		idMap[node.ID] = i
	}
	nc := len(g.Nodes)
	n := &refNetwork{
		values:   make([]float64, nc),
		cell:     make([]float64, nc),
		kind:     make([]NodeType, nc),
		off:      make([]int32, nc*int(NumGates)+1),
		out:      make([]float64, g.NumOutputs),
		inStart:  1,
		inEnd:    1 + g.NumInputs,
		outStart: 1 + g.NumInputs,
		outEnd:   1 + g.NumInputs + g.NumOutputs,
	}
	for i, node := range g.Nodes {
		n.kind[i] = node.Type
	}

	// Count then fill connections per (target, gate) slot.
	type edge struct {
		from, slot int
		w, hebb    float64
	}
	var edges []edge
	deps := make([][]int, nc)
	for _, c := range g.Conns {
		fi, okIn := idMap[c.In]
		ti, okOut := idMap[c.Out]
		if !c.Enabled || !okIn || !okOut {
			continue
		}
		gate := c.Gate
		if n.kind[ti] != Memory || gate >= NumGates {
			gate = GateIn
		}
		slot := ti*int(NumGates) + int(gate)
		edges = append(edges, edge{fi, slot, c.Weight, c.Hebb})
		n.off[slot+1]++
		deps[ti] = append(deps[ti], fi)
	}
	for i := 1; i < len(n.off); i++ {
		n.off[i] += n.off[i-1]
	}
	n.from = make([]int32, len(edges))
	n.weight = make([]float64, len(edges))
	fill := make([]int32, len(n.off))
	copy(fill, n.off)
	for _, e := range edges {
		k := fill[e.slot]
		n.from[k], n.weight[k] = int32(e.from), e.w
		if e.hebb != 0 {
			n.plastic = append(n.plastic, refPlasticLink{k, int32(e.from), int32(e.slot / int(NumGates)), e.hebb})
		}
		fill[e.slot]++
	}

	// Topological order; cycles are broken (a back edge reads the value
	// computed so far in this tick, or the memory node's previous output).
	visited := make([]bool, nc)
	for i := 0; i < n.inEnd; i++ {
		visited[i] = true
	}
	inStack := make([]bool, nc)
	var visit func(int)
	visit = func(node int) {
		if visited[node] || inStack[node] {
			return
		}
		inStack[node] = true
		for _, d := range deps[node] {
			visit(d)
		}
		inStack[node] = false
		visited[node] = true
		n.order = append(n.order, int32(node))
	}
	for i := range nc {
		visit(i)
	}
	return n
}

// Activate runs one tick. Plain nodes are recomputed from scratch; Memory
// nodes carry their cell over, which gives the ant a short-term memory.
// The returned slice is owned by the network and reused on the next call.
func (n *refNetwork) Activate(inputs []float64) []float64 {
	v := n.values
	v[0] = 1.0
	for i := n.inStart; i < n.inEnd; i++ {
		v[i] = 0
		if k := i - n.inStart; k < len(inputs) {
			v[i] = inputs[k]
		}
	}
	for _, ni := range n.order {
		base := int(ni) * int(NumGates)
		if n.kind[ni] != Memory {
			v[ni] = refSigmoid(n.sum(base))
			continue
		}
		// LSTM cell. Gates default to "mostly open" (bias 2) so a fresh memory
		// node behaves like a leaky integrator until evolution wires its gates.
		x := n.sum(base + int(GateIn))
		ig := refSigmoid01(n.sum(base+int(GateInput)) + 2)
		fg := refSigmoid01(n.sum(base+int(GateForget)) + 2)
		og := refSigmoid01(n.sum(base+int(GateOutput)) + 2)
		c := fg*n.cell[ni] + ig*refFastTanh(x)
		n.cell[ni] = c
		v[ni] = og * refFastTanh(c)
	}
	copy(n.out, v[n.outStart:n.outEnd])
	n.learn()
	return n.out
}

// learn applies Oja's rule to the plastic links: Hebbian (links between
// neurons that fire together strengthen) yet bounded, as the decay term
// pulls the weight back when the target fires a lot. What an ant learns
// this way dies with her: only the plasticity itself is inherited.
func (n *refNetwork) learn() {
	v := n.values
	for _, p := range n.plastic {
		pre, post := v[p.from], v[p.to]
		w := n.weight[p.k] + p.eta*post*(pre-post*n.weight[p.k])
		n.weight[p.k] = max(-8, min(8, w))
	}
}

func (n *refNetwork) sum(slot int) float64 {
	s := 0.0
	v := n.values
	for k := n.off[slot]; k < n.off[slot+1]; k++ {
		s += v[n.from[k]] * n.weight[k]
	}
	return s
}

// Value returns the last activation of the node at index i (genome node order).
func (n *refNetwork) Value(i int) float64 {
	if i < 0 || i >= len(n.values) {
		return 0
	}
	return n.values[i]
}

// Cell returns the memory cell of node i (0 for plain nodes).
func (n *refNetwork) Cell(i int) float64 {
	if i < 0 || i >= len(n.cell) {
		return 0
	}
	return n.cell[i]
}

// refSigmoid is the steepened NEAT sigmoid 1/(1+e^(-4.9x)), computed through a
// rational tanh approximation instead of math.Exp: about 4x faster, with
// an absolute error below 0.012.
func refSigmoid(x float64) float64 { return 0.5 + 0.5*refFastTanh(2.45*x) }

func refSigmoid01(x float64) float64 { return 0.5 + 0.5*refFastTanh(0.5*x) }

// refFastTanh is a Padé approximation of tanh, exact at 0 and saturating at ±3.
func refFastTanh(x float64) float64 {
	if x > 3 {
		return 1
	}
	if x < -3 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}
