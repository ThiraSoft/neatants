package neat

// Flat is a compiled network as plain arrays, its nodes sorted by level: a
// node only reads, from this tick, nodes of lower levels (inputs and bias are
// level 0), so the nodes of one level can be computed in any order or all at
// once. It is the form the GPU evaluator uploads; Activate below runs the
// same algorithm on the CPU, which is how the GPU is checked.
type Flat struct {
	Kind                             []uint8   // NodeType per node
	Order                            []int32   // computed nodes, sorted by level
	LevelStart                       []int32   // level l+1 is Order[LevelStart[l]:LevelStart[l+1]]
	Off                              []int32   // per (node, gate) slot: Off[s]..Off[s+1] into From/Weight
	From                             []int32   // index into the 2n values; >= n reads the previous tick
	Weight                           []float32 // starting weights (plasticity changes them during the life)
	Plastic                          []Plastic
	InStart, InEnd, OutStart, OutEnd int32

	// The interpreter's views of Kind and Plastic, built once by Network.Flat.
	kindCache    []NodeType
	plasticCache []plasticLink
}

// Plastic is a link whose weight learns (Oja's rule) during the life.
type Plastic struct {
	K, From, To int32
	Eta         float32
}

// FlatState is what changes while a flat network lives.
type FlatState struct {
	Vals   []float32 // 2n: this tick, then the previous one
	Cell   []float32
	Weight []float32
	Out    []float32
}

// Flat returns the network as it is now (call it before the first Activate
// to get a fresh network).
func (n *Network) Flat() *Flat {
	f := &Flat{
		Kind:     make([]uint8, n.n),
		Off:      append([]int32(nil), n.off...),
		From:     append([]int32(nil), n.from...),
		Weight:   append([]float32(nil), n.weight...),
		InStart:  int32(n.inStart),
		InEnd:    int32(n.inEnd),
		OutStart: int32(n.outStart),
		OutEnd:   int32(n.outEnd),
	}
	for i, k := range n.kind {
		f.Kind[i] = uint8(k)
	}
	for _, p := range n.plastic {
		f.Plastic = append(f.Plastic, Plastic{p.k, p.from, p.to, p.eta})
	}
	// order is topological for this tick's edges, so one pass sets levels.
	level := make([]int32, n.n)
	for _, ni := range n.order {
		var l int32
		for s := int(ni) * int(NumGates); s < (int(ni)+1)*int(NumGates); s++ {
			for k := n.off[s]; k < n.off[s+1]; k++ {
				if src := n.from[k]; int(src) < n.n {
					l = max(l, level[src])
				}
			}
		}
		level[ni] = l + 1
	}
	// A stable counting sort of the order by level: the same result as a
	// stable sort, without its cost on networks of a thousand nodes.
	top := int32(0)
	for _, ni := range n.order {
		top = max(top, level[ni])
	}
	count := make([]int32, top+2)
	for _, ni := range n.order {
		count[level[ni]+1]++
	}
	for l := 1; l < len(count); l++ {
		count[l] += count[l-1]
	}
	f.Order = make([]int32, len(n.order))
	at := append([]int32(nil), count...)
	for _, ni := range n.order {
		f.Order[at[level[ni]]] = ni
		at[level[ni]]++
	}
	// Levels start at 1 (inputs and bias are 0), so count[1:] are the
	// starts of the non-empty levels, every level from 1 to top having a
	// node by construction.
	f.LevelStart = []int32{0}
	for l := int32(1); l <= top; l++ {
		f.LevelStart = append(f.LevelStart, count[l+1])
	}
	f.kindCache = append([]NodeType(nil), n.kind...)
	f.plasticCache = append([]plasticLink(nil), n.plastic...)
	return f
}

// Nodes is the number of nodes, inputs and bias included.
func (f *Flat) Nodes() int { return len(f.Kind) }

// NewState returns the state of a newborn network.
func (f *Flat) NewState() *FlatState {
	return &FlatState{
		Vals:   make([]float32, 2*len(f.Kind)),
		Cell:   make([]float32, len(f.Kind)),
		Weight: append([]float32(nil), f.Weight...),
		Out:    make([]float32, f.OutEnd-f.OutStart),
	}
}

// Activate runs one tick level by level, the way the GPU kernel does, and
// gives bit for bit what Network.Activate gives.
func (f *Flat) Activate(s *FlatState, inputs []float32) []float32 {
	n := len(f.Kind)
	v := s.Vals
	v[0] = 1.0
	for i := f.InStart; i < f.InEnd; i++ {
		v[i] = 0
		if k := int(i - f.InStart); k < len(inputs) {
			v[i] = inputs[k]
		}
	}
	copy(v[n:], v[:n])
	for l := 0; l+1 < len(f.LevelStart); l++ {
		for _, ni := range f.Order[f.LevelStart[l]:f.LevelStart[l+1]] {
			evalNode(v, s.Cell, f.kindCache, f.Off, f.From, s.Weight, ni)
		}
	}
	copy(s.Out, v[f.OutStart:f.OutEnd])
	learn(v, s.Weight, f.plasticCache)
	return s.Out
}

// EdgeGenes returns, for each edge of the network g builds (the From and
// Weight index of its Flat), the index in g.Conns of the gene it comes from,
// so that a learner can write the weights it found back into the genome.
func (g *Genome) EdgeGenes() []int32 {
	idx := newIDIndex(g.Nodes)
	defer idx.release()
	nc := len(g.Nodes)
	kind := make([]NodeType, nc)
	for i, node := range g.Nodes {
		kind[i] = node.Type
	}
	// The same slots and the same fill order as BuildNetwork.
	off := make([]int32, nc*int(NumGates)+1)
	slots := make([]int, len(g.Conns))
	for i, c := range g.Conns {
		slots[i] = -1
		_, okIn := idx.get(c.In)
		ti, okOut := idx.get(c.Out)
		if !c.Enabled || !okIn || !okOut {
			continue
		}
		gate := c.Gate
		if kind[ti] != Memory || gate >= NumGates {
			gate = GateIn
		}
		slots[i] = ti*int(NumGates) + int(gate)
		off[slots[i]+1]++
	}
	for i := 1; i < len(off); i++ {
		off[i] += off[i-1]
	}
	genes := make([]int32, off[len(off)-1])
	for i, s := range slots {
		if s < 0 {
			continue
		}
		genes[off[s]] = int32(i)
		off[s]++
	}
	return genes
}
