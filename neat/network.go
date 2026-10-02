package neat

// Gates of a Memory node. Plain nodes only use GateIn.
const (
	GateIn     uint8 = iota // candidate value
	GateInput               // how much of the candidate is written
	GateForget              // how much of the previous cell is kept
	GateOutput              // how much of the cell is exposed
	NumGates
)

type plasticLink struct {
	k, from, to int32
	eta         float32
}

// Network is the compiled, flat form of a genome. Incoming connections are
// stored contiguously per (node, gate) so activation walks linear memory.
//
// vals holds two copies of the node values: [0, n) is this tick, [n, 2n) is
// the previous one. An edge that closes a cycle (its source comes at or after
// its target in the activation order) reads the previous tick, so from[k]
// points into the second half for it. That is what the network always did,
// implicitly; making it explicit lets the nodes of one level be computed in
// any order, which is what the GPU does.
type Network struct {
	vals   []float32
	cell   []float32 // LSTM cell state, persists between ticks
	kind   []NodeType
	order  []int32
	off    []int32 // off[n*NumGates+g] .. off[n*NumGates+g+1] index into from/weight
	from   []int32 // index into vals
	weight []float32
	out    []float32
	// plastic lists the links whose weight learns during the life (Hebb != 0).
	plastic []plasticLink

	n                                int
	inStart, inEnd, outStart, outEnd int

	// Tag is free for an evaluator that keeps its own copy of the network
	// (the GPU arena keeps its slot here). BuildNetwork sets it to -1.
	Tag int32
}

func (g *Genome) BuildNetwork() *Network {
	idMap := make(map[int]int, len(g.Nodes))
	for i, node := range g.Nodes {
		idMap[node.ID] = i
	}
	nc := len(g.Nodes)
	n := &Network{
		vals:     make([]float32, 2*nc),
		cell:     make([]float32, nc),
		kind:     make([]NodeType, nc),
		off:      make([]int32, nc*int(NumGates)+1),
		out:      make([]float32, g.NumOutputs),
		n:        nc,
		inStart:  1,
		inEnd:    1 + g.NumInputs,
		outStart: 1 + g.NumInputs,
		outEnd:   1 + g.NumInputs + g.NumOutputs,
		Tag:      -1,
	}
	for i, node := range g.Nodes {
		n.kind[i] = node.Type
	}

	// Count then fill connections per (target, gate) slot.
	type edge struct {
		from, slot int
		w, hebb    float64
	}
	edges := make([]edge, 0, len(g.Conns))
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

	// Topological order; cycles are broken (a back edge reads the previous
	// tick, see Network).
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
	// pos is each node's place in the order; inputs and bias come first.
	pos := make([]int, nc)
	for i := range pos {
		pos[i] = -1
	}
	for p, ni := range n.order {
		pos[ni] = p
	}

	n.from = make([]int32, len(edges))
	n.weight = make([]float32, len(edges))
	fill := make([]int32, len(n.off))
	copy(fill, n.off)
	for _, e := range edges {
		k := fill[e.slot]
		to := e.slot / int(NumGates)
		src := int32(e.from)
		if pos[e.from] >= pos[to] && pos[to] >= 0 {
			src += int32(nc) // back edge: previous tick
		}
		n.from[k], n.weight[k] = src, float32(e.w)
		if e.hebb != 0 {
			n.plastic = append(n.plastic, plasticLink{k, int32(e.from), int32(to), float32(e.hebb)})
		}
		fill[e.slot]++
	}
	return n
}

// Activate runs one tick. Plain nodes are recomputed from scratch; Memory
// nodes carry their cell over, which gives the ant a short-term memory.
// The returned slice is owned by the network and reused on the next call.
func (n *Network) Activate(inputs []float32) []float32 {
	v := n.vals
	v[0] = 1.0
	for i := n.inStart; i < n.inEnd; i++ {
		v[i] = 0
		if k := i - n.inStart; k < len(inputs) {
			v[i] = inputs[k]
		}
	}
	copy(v[n.n:], v[:n.n])
	for _, ni := range n.order {
		evalNode(v, n.cell, n.kind, n.off, n.from, n.weight, ni)
	}
	copy(n.out, v[n.outStart:n.outEnd])
	learn(v, n.weight, n.plastic)
	return n.out
}

// evalNode computes node ni from the values it reads. The CPU network, the
// flat interpreter and the GPU kernel all do exactly this.
func evalNode(v, cell []float32, kind []NodeType, off, from []int32, weight []float32, ni int32) {
	base := int(ni) * int(NumGates)
	if kind[ni] != Memory {
		v[ni] = sigmoid32(sum(v, off, from, weight, base))
		return
	}
	// LSTM cell. Gates default to "mostly open" (bias 2) so a fresh memory
	// node behaves like a leaky integrator until evolution wires its gates.
	x := sum(v, off, from, weight, base+int(GateIn))
	ig := sigmoid01_32(sum(v, off, from, weight, base+int(GateInput)) + 2)
	fg := sigmoid01_32(sum(v, off, from, weight, base+int(GateForget)) + 2)
	og := sigmoid01_32(sum(v, off, from, weight, base+int(GateOutput)) + 2)
	c := fg*cell[ni] + ig*fastTanh32(x)
	cell[ni] = c
	v[ni] = og * fastTanh32(c)
}

// learn applies Oja's rule to the plastic links: Hebbian (links between
// neurons that fire together strengthen) yet bounded, as the decay term
// pulls the weight back when the target fires a lot. What an ant learns
// this way dies with her: only the plasticity itself is inherited.
func learn(v, weight []float32, plastic []plasticLink) {
	for _, p := range plastic {
		pre, post := v[p.from], v[p.to]
		w := weight[p.k] + p.eta*post*(pre-post*weight[p.k])
		weight[p.k] = max(-8, min(8, w))
	}
}

func sum(v []float32, off, from []int32, weight []float32, slot int) float32 {
	var s float32
	for k := off[slot]; k < off[slot+1]; k++ {
		s += v[from[k]] * weight[k]
	}
	return s
}

// Value returns the last activation of the node at index i (genome node order).
func (n *Network) Value(i int) float64 {
	if i < 0 || i >= n.n {
		return 0
	}
	return float64(n.vals[i])
}

// Cell returns the memory cell of node i (0 for plain nodes).
func (n *Network) Cell(i int) float64 {
	if i < 0 || i >= len(n.cell) {
		return 0
	}
	return float64(n.cell[i])
}

// sigmoid32 is the steepened NEAT sigmoid 1/(1+e^(-4.9x)), computed through a
// rational tanh approximation instead of an exponential: about 4x faster, with
// an absolute error below 0.012.
func sigmoid32(x float32) float32 { return 0.5 + 0.5*fastTanh32(2.45*x) }

func sigmoid01_32(x float32) float32 { return 0.5 + 0.5*fastTanh32(0.5*x) }

// fastTanh32 is a Padé approximation of tanh, exact at 0 and saturating at ±3.
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
