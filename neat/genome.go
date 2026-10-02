package neat

import (
	"cmp"
	"math"
	"math/rand"
	"slices"
	"sync"
)

var (
	// histMu guards the innovation and node histories: several worlds can
	// evolve at once in the same process and must share the same numbering.
	histMu sync.Mutex

	innovationNum     int
	innovationHistory map[[3]int]int

	// Node IDs are global like innovations: the same structural mutation
	// gives the same node ID in every genome, so crossover can tell two
	// neurons apart.
	nodeNum     int
	nodeHistory map[[3]int]int
)

func init() {
	innovationHistory = make(map[[3]int]int)
	nodeHistory = make(map[[3]int]int)
}

const (
	nodeSplit  = iota // hidden node splitting a connection
	nodeMemory        // memory cell fed by one node, read by another
)

// globalNodeID returns the ID registered for this structural mutation, or a
// fresh one when g already holds it (same connection split twice).
func (g *Genome) globalNodeID(key [3]int) int {
	histMu.Lock()
	defer histMu.Unlock()
	id, ok := nodeHistory[key]
	if !ok {
		nodeNum++
		id = nodeNum
		nodeHistory[key] = id
	}
	for _, n := range g.Nodes {
		if n.ID == id {
			nodeNum++
			return nodeNum
		}
	}
	return id
}

func getInnovation(in, out int) int { return getGateInnovation(in, out, GateIn) }

func getGateInnovation(in, out int, gate uint8) int {
	key := [3]int{in, out, int(gate)}
	histMu.Lock()
	defer histMu.Unlock()
	if inn, ok := innovationHistory[key]; ok {
		return inn
	}
	innovationNum++
	innovationHistory[key] = innovationNum
	return innovationNum
}

// SyncInnovations registers the innovations of loaded genomes so new
// mutations never reuse their numbers (which would corrupt crossover).
func SyncInnovations(genomes []*Genome) {
	histMu.Lock()
	defer histMu.Unlock()
	for _, g := range genomes {
		for _, c := range g.Conns {
			key := [3]int{c.In, c.Out, int(c.Gate)}
			if _, ok := innovationHistory[key]; !ok {
				innovationHistory[key] = c.Innovation
			}
			innovationNum = max(innovationNum, c.Innovation)
		}
		for _, n := range g.Nodes {
			nodeNum = max(nodeNum, n.ID)
		}
	}
}

type NodeType byte

const (
	Sensor NodeType = iota
	Hidden
	Output
	Bias
	Memory // LSTM cell: keeps a state between ticks, driven by gated inputs
)

type NodeGene struct {
	ID   int      `json:"id"`
	Type NodeType `json:"type"`
}

type ConnGene struct {
	In         int     `json:"in"`
	Out        int     `json:"out"`
	Weight     float64 `json:"weight"`
	Enabled    bool    `json:"enabled"`
	Innovation int     `json:"innovation"`
	Gate       uint8   `json:"gate,omitempty"` // target gate when Out is a Memory node
	// Hebb is the evolved plasticity of the link: during a life its weight
	// follows Oja's rule, Δw = Hebb·post·(pre − post·w). Zero keeps it fixed.
	Hebb float64 `json:"hebb,omitempty"`
}

type Genome struct {
	Nodes      []NodeGene `json:"nodes"`
	Conns      []ConnGene `json:"conns"`
	Fitness    float64    `json:"fitness"`
	ID         int        `json:"id"`
	NumInputs  int        `json:"num_inputs"`
	NumOutputs int        `json:"num_outputs"`
	// Traits holds heritable non-neural genes in [0,1] (elements, morphology...).
	Traits []float64 `json:"traits,omitempty"`
	// MutPower is the genome's own weight-mutation strength (self-adaptive,
	// as in evolution strategies). Zero means "not set yet".
	MutPower float64 `json:"mut_power,omitempty"`
	// Origin is the ID of the genome this one is an unchanged copy of (zero
	// once mutated), and Evals the number of lives its Fitness averages, so a
	// cloned champion is judged on several lives rather than one lucky run.
	Origin int `json:"origin,omitempty"`
	Evals  int `json:"evals,omitempty"`

	sorted []innWeight // innovations and weights sorted by innovation, cached for Compatibility
}

// Advanced switches on the modern mutation scheme: self-adaptive strength,
// sparse weight perturbation and connection toggling. Off keeps the classic
// NEAT defaults.
var Advanced = true

const defaultMutPower = 0.3

// DirectLinkRate is the share of input → output links in a fresh genome with
// hidden nodes. A genome without any starts as in FS-NEAT: each output reads
// MinimalLinks random inputs (plus the bias), and evolution picks the inputs
// that matter instead of tuning a weight for every one of them.
var (
	DirectLinkRate = 0.06
	MinimalLinks   = 1
)

// WeightsPerMutation is the number of weights a mutation perturbs on average,
// whatever the size of the network: a fixed share would scramble big ones.
var WeightsPerMutation = 3.0

// Lineage identifies a genome across its unchanged copies.
func (g *Genome) Lineage() int {
	if g.Origin > 0 {
		return g.Origin
	}
	return g.ID
}

// RandomTraits returns n uniformly random traits in [0,1].
func RandomTraits(n int) []float64 {
	t := make([]float64, n)
	for i := range t {
		t[i] = rand.Float64()
	}
	return t
}

func NewGenome(id, inputs, outputs int) *Genome {
	return NewGenomeWithHidden(id, inputs, outputs, 0)
}

func NewGenomeWithHidden(id, inputs, outputs, hidden int) *Genome {
	g := &Genome{
		ID:         id,
		NumInputs:  inputs,
		NumOutputs: outputs,
	}
	nid := 0
	g.Nodes = append(g.Nodes, NodeGene{nid, Bias})
	nid++
	for i := 0; i < inputs; i++ {
		g.Nodes = append(g.Nodes, NodeGene{nid, Sensor})
		nid++
	}
	firstOutput := nid
	for i := 0; i < outputs; i++ {
		g.Nodes = append(g.Nodes, NodeGene{nid, Output})
		nid++
	}
	firstHidden := nid
	for i := 0; i < hidden; i++ {
		g.Nodes = append(g.Nodes, NodeGene{nid, Hidden})
		nid++
	}
	histMu.Lock()
	nodeNum = max(nodeNum, nid-1) // the base layout is shared, mutations start above it
	histMu.Unlock()

	// Connect inputs → hidden (~15% sparse)
	for h := 0; h < hidden; h++ {
		hid := firstHidden + h
		// Bias → hidden
		if rand.Float64() < 0.3 {
			g.Conns = append(g.Conns, ConnGene{
				In: 0, Out: hid,
				Weight: rand.Float64()*2 - 1, Enabled: true,
				Innovation: getInnovation(0, hid),
			})
		}
		for i := 1; i <= inputs; i++ {
			if rand.Float64() < 0.15 {
				g.Conns = append(g.Conns, ConnGene{
					In: i, Out: hid,
					Weight: rand.Float64()*4 - 2, Enabled: true,
					Innovation: getInnovation(i, hid),
				})
			}
		}
	}

	// Connect hidden → outputs (~25% sparse), plus a few direct input →
	// output links as in the original NEAT, so simple reflexes are one
	// mutation away.
	for j := 0; j < outputs; j++ {
		out := firstOutput + j
		var reads []int
		if hidden == 0 {
			reads = rand.Perm(inputs)[:min(MinimalLinks, inputs)]
		} else {
			for i := range inputs {
				if rand.Float64() < DirectLinkRate {
					reads = append(reads, i)
				}
			}
		}
		for _, k := range reads {
			i := k + 1
			g.Conns = append(g.Conns, ConnGene{
				In: i, Out: out,
				Weight: rand.Float64()*4 - 2, Enabled: true,
				Innovation: getInnovation(i, out),
			})
		}
		// Bias → output
		g.Conns = append(g.Conns, ConnGene{
			In: 0, Out: out,
			Weight: rand.Float64()*2 - 1, Enabled: true,
			Innovation: getInnovation(0, out),
		})
		for h := 0; h < hidden; h++ {
			hid := firstHidden + h
			if rand.Float64() < 0.25 {
				g.Conns = append(g.Conns, ConnGene{
					In: hid, Out: out,
					Weight: rand.Float64()*4 - 2, Enabled: true,
					Innovation: getInnovation(hid, out),
				})
			}
		}
	}

	// Some hidden → hidden connections (~5% sparse, feed-forward only: lower → higher)
	for h1 := 0; h1 < hidden; h1++ {
		for h2 := h1 + 1; h2 < hidden; h2++ {
			if rand.Float64() < 0.05 {
				g.Conns = append(g.Conns, ConnGene{
					In: firstHidden + h1, Out: firstHidden + h2,
					Weight: rand.Float64()*4 - 2, Enabled: true,
					Innovation: getInnovation(firstHidden+h1, firstHidden+h2),
				})
			}
		}
	}

	return g
}

func (g *Genome) Copy() *Genome {
	c := &Genome{
		ID: g.ID, Fitness: 0,
		NumInputs: g.NumInputs, NumOutputs: g.NumOutputs,
		MutPower: g.MutPower, Origin: g.Origin, Evals: g.Evals,
	}
	c.Nodes = make([]NodeGene, len(g.Nodes))
	copy(c.Nodes, g.Nodes)
	c.Conns = make([]ConnGene, len(g.Conns))
	copy(c.Conns, g.Conns)
	if g.Traits != nil {
		c.Traits = make([]float64, len(g.Traits))
		copy(c.Traits, g.Traits)
	}
	return c
}

func (g *Genome) Mutate() {
	g.sorted = nil
	g.Origin, g.Evals = 0, 0
	if Advanced {
		g.mutateWeightsAdaptive()
	} else if rand.Float64() < 0.8 {
		for i := range g.Conns {
			if rand.Float64() < 0.9 {
				g.Conns[i].Weight += rand.NormFloat64() * 0.3
				g.Conns[i].Weight = clamp(g.Conns[i].Weight, -8, 8)
			} else {
				g.Conns[i].Weight = rand.Float64()*4 - 2
			}
		}
	}
	if Advanced && rand.Float64() < 0.015 && len(g.Conns) > 0 {
		// Toggle a connection: lets networks prune as well as grow.
		i := rand.Intn(len(g.Conns))
		g.Conns[i].Enabled = !g.Conns[i].Enabled
	}
	for i := range g.Traits {
		if rand.Float64() < 0.25 {
			g.Traits[i] = clamp(g.Traits[i]+rand.NormFloat64()*0.06, 0, 1)
		}
	}
	if rand.Float64() < HebbRate && len(g.Conns) > 0 {
		i := rand.Intn(len(g.Conns))
		g.Conns[i].Hebb = clamp(g.Conns[i].Hebb+rand.NormFloat64()*0.002, -MaxHebb, MaxHebb)
	}
	if rand.Float64() < 0.05 {
		g.addConnMutation()
	}
	if rand.Float64() < AddNodeRate {
		g.addNodeMutation()
	}
	if rand.Float64() < AddMemoryRate {
		g.addMemoryMutation()
	}
	if rand.Float64() < RemoveNodeRate {
		g.removeNodeMutation()
	}
	g.prune()
}

// Structural rates, per Mutate. Removal nearly balances the additions, so a
// neuron only stays if selection keeps it: networks grew without bound (and
// without any fitness gain) when only dead neurons were ever pruned.
var (
	AddNodeRate    = 0.015
	AddMemoryRate  = 0.01
	RemoveNodeRate = 0.02
)

// maxBypass caps the links a removal adds to keep the signal flowing.
const maxBypass = 4

// removeNodeMutation deletes a random hidden or memory neuron. When it has
// few enough inputs and outputs, each input is wired straight to each output
// with the product of the two weights, as in SharpNEAT's simplification, so
// the removal disturbs the network as little as possible.
func (g *Genome) removeNodeMutation() {
	var cands []int
	for _, n := range g.Nodes {
		if n.Type == Hidden || n.Type == Memory {
			cands = append(cands, n.ID)
		}
	}
	if len(cands) == 0 {
		return
	}
	id := cands[rand.Intn(len(cands))]
	var ins, outs []ConnGene
	for _, c := range g.Conns {
		if !c.Enabled || c.In == c.Out {
			continue
		}
		if c.Out == id && c.Gate == GateIn {
			ins = append(ins, c)
		} else if c.In == id {
			outs = append(outs, c)
		}
	}
	conns := g.Conns[:0]
	for _, c := range g.Conns {
		if c.In != id && c.Out != id {
			conns = append(conns, c)
		}
	}
	g.Conns = conns
	nodes := g.Nodes[:0]
	for _, n := range g.Nodes {
		if n.ID != id {
			nodes = append(nodes, n)
		}
	}
	g.Nodes = nodes
	if len(ins)*len(outs) > maxBypass {
		return
	}
	for _, a := range ins {
		for _, b := range outs {
			if a.In == b.Out {
				continue
			}
			w := clamp(a.Weight*b.Weight, -8, 8)
			found := false
			for i, c := range g.Conns {
				if c.In == a.In && c.Out == b.Out && c.Gate == b.Gate {
					g.Conns[i].Weight = clamp(c.Weight+w, -8, 8)
					g.Conns[i].Enabled = true
					found = true
					break
				}
			}
			if !found {
				g.Conns = append(g.Conns, ConnGene{
					In: a.In, Out: b.Out, Weight: w, Enabled: true,
					Innovation: getGateInnovation(a.In, b.Out, b.Gate), Gate: b.Gate,
				})
			}
		}
	}
}

// HebbRate is the chance per Mutate to change the plasticity of one link.
var (
	HebbRate = 0.05
	MaxHebb  = 0.02
)

// Pruning rates, per Mutate: each disabled connection may be forgotten, and
// each dead neuron (nothing reads it) may be removed with its connections.
var (
	PruneConnRate = 0.05
	PruneNodeRate = 0.2
)

// prune lets networks shrink: without it structure only ever accumulates.
// A neuron with no enabled outgoing link has no effect on the outputs. One
// without inputs still is a constant (bias-like) source, so it is kept.
func (g *Genome) prune() {
	reads := make(map[int]bool, len(g.Nodes))
	for _, c := range g.Conns {
		if c.Enabled && c.In != c.Out {
			reads[c.In] = true
		}
	}
	dead := map[int]bool{}
	nodes := g.Nodes[:0]
	for _, n := range g.Nodes {
		if (n.Type == Hidden || n.Type == Memory) && !reads[n.ID] && rand.Float64() < PruneNodeRate {
			dead[n.ID] = true
			continue
		}
		nodes = append(nodes, n)
	}
	g.Nodes = nodes
	conns := g.Conns[:0]
	for _, c := range g.Conns {
		if dead[c.In] || dead[c.Out] || (!c.Enabled && rand.Float64() < PruneConnRate) {
			continue
		}
		conns = append(conns, c)
	}
	g.Conns = conns
}

// addMemoryMutation inserts an LSTM cell fed by one node and read by another,
// sometimes with its forget gate already driven by a third.
// AddMemory inserts one LSTM cell (see addMemoryMutation).
func (g *Genome) AddMemory() { g.addMemoryMutation() }

func (g *Genome) addMemoryMutation() {
	var srcs, dsts []int
	for _, n := range g.Nodes {
		if n.Type != Output {
			srcs = append(srcs, n.ID)
		}
		if n.Type == Output || n.Type == Hidden || n.Type == Memory {
			dsts = append(dsts, n.ID)
		}
	}
	if len(srcs) == 0 || len(dsts) == 0 {
		return
	}
	in := srcs[rand.Intn(len(srcs))]
	out := dsts[rand.Intn(len(dsts))]
	nid := g.globalNodeID([3]int{nodeMemory, in, out})
	g.Nodes = append(g.Nodes, NodeGene{nid, Memory})
	g.Conns = append(g.Conns,
		ConnGene{In: in, Out: nid, Weight: rand.Float64()*4 - 2, Enabled: true, Innovation: getGateInnovation(in, nid, GateIn)},
		ConnGene{In: nid, Out: out, Weight: rand.Float64()*4 - 2, Enabled: true, Innovation: getInnovation(nid, out)},
	)
	if rand.Float64() < 0.5 {
		f := srcs[rand.Intn(len(srcs))]
		g.Conns = append(g.Conns, ConnGene{In: f, Out: nid, Weight: rand.Float64()*4 - 2, Enabled: true,
			Innovation: getGateInnovation(f, nid, GateForget), Gate: GateForget})
	}
}

// mutateWeightsAdaptive first mutates the mutation strength itself (log-normal
// self-adaptation), then perturbs only a fraction of the weights with it, so
// a good network is refined rather than scrambled.
func (g *Genome) mutateWeightsAdaptive() {
	p := g.MutPower
	if p == 0 {
		p = defaultMutPower
	}
	p = clamp(p*math.Exp(0.2*rand.NormFloat64()), 0.02, 1.5)
	g.MutPower = p
	if len(g.Conns) == 0 {
		return
	}
	rate := math.Min(1, WeightsPerMutation/float64(len(g.Conns)))
	for i := range g.Conns {
		if rand.Float64() >= rate {
			continue
		}
		if rand.Float64() < 0.95 {
			g.Conns[i].Weight = clamp(g.Conns[i].Weight+rand.NormFloat64()*p, -8, 8)
		} else {
			g.Conns[i].Weight = rand.Float64()*4 - 2
		}
	}
}

func (g *Genome) addConnMutation() {
	for tries := 0; tries < 30; tries++ {
		a := g.Nodes[rand.Intn(len(g.Nodes))]
		b := g.Nodes[rand.Intn(len(g.Nodes))]
		if a.ID == b.ID || b.Type == Sensor || b.Type == Bias || a.Type == Output {
			continue
		}
		gate := GateIn
		if b.Type == Memory {
			gate = uint8(rand.Intn(int(NumGates)))
		}
		exists := false
		for _, c := range g.Conns {
			if c.In == a.ID && c.Out == b.ID && c.Gate == gate {
				exists = true
				break
			}
		}
		if exists {
			continue
		}
		g.Conns = append(g.Conns, ConnGene{
			In: a.ID, Out: b.ID,
			Weight: rand.Float64()*4 - 2, Enabled: true,
			Innovation: getGateInnovation(a.ID, b.ID, gate), Gate: gate,
		})
		return
	}
}

func (g *Genome) addNodeMutation() {
	enabled := []int{}
	for i, c := range g.Conns {
		if c.Enabled {
			enabled = append(enabled, i)
		}
	}
	if len(enabled) == 0 {
		return
	}
	ci := enabled[rand.Intn(len(enabled))]
	g.Conns[ci].Enabled = false
	old := g.Conns[ci]
	nid := g.globalNodeID([3]int{nodeSplit, old.Innovation, 0})
	g.Nodes = append(g.Nodes, NodeGene{nid, Hidden})
	g.Conns = append(g.Conns,
		ConnGene{In: old.In, Out: nid, Weight: 1, Enabled: true, Innovation: getInnovation(old.In, nid)},
		ConnGene{In: nid, Out: old.Out, Weight: old.Weight, Enabled: true, Innovation: getGateInnovation(nid, old.Out, old.Gate), Gate: old.Gate},
	)
}

func Crossover(better, other *Genome, childID int) *Genome {
	child := better.Copy()
	child.ID = childID
	if better.MutPower > 0 && other.MutPower > 0 {
		child.MutPower = math.Sqrt(better.MutPower * other.MutPower)
	}
	if len(child.Traits) == len(other.Traits) {
		for i := range child.Traits {
			t := rand.Float64()
			child.Traits[i] = child.Traits[i]*t + other.Traits[i]*(1-t)
		}
	}
	om := make(map[int]ConnGene, len(other.Conns))
	for _, c := range other.Conns {
		om[c.Innovation] = c
	}
	// A matching gene comes whole from one parent, weight and state: the
	// classic "disabled in either parent → 75 % disabled" rule ratchets
	// connections off generation after generation.
	for i, c := range child.Conns {
		if oc, ok := om[c.Innovation]; ok && rand.Float64() < 0.5 {
			child.Conns[i].Weight = oc.Weight
			child.Conns[i].Enabled = oc.Enabled
		}
	}
	child.importGenes(other)
	return child
}

// ImportRate is the chance that a gene only the weaker parent carries is
// passed on too, so a child inherits structure from both lineages.
var ImportRate = 0.25

// importGenes copies some of other's enabled disjoint and excess connections, adding
// the nodes they need. Node IDs are global, but genomes saved before that had
// local ones: a gene whose node ID exists here with another type belongs to a
// different neuron and is skipped.
func (g *Genome) importGenes(other *Genome) {
	have := make(map[int]bool, len(g.Conns))
	for _, c := range g.Conns {
		have[c.Innovation] = true
	}
	types := make(map[int]NodeType, len(g.Nodes))
	for _, n := range g.Nodes {
		types[n.ID] = n.Type
	}
	otypes := make(map[int]NodeType, len(other.Nodes))
	for _, n := range other.Nodes {
		otypes[n.ID] = n.Type
	}
	fits := func(id int) bool {
		ot, ok := otypes[id]
		if !ok {
			return false
		}
		t, mine := types[id]
		return !mine || t == ot
	}
	for _, oc := range other.Conns {
		if !oc.Enabled || have[oc.Innovation] || rand.Float64() >= ImportRate || !fits(oc.In) || !fits(oc.Out) {
			continue
		}
		for _, id := range [2]int{oc.In, oc.Out} {
			if _, ok := types[id]; !ok {
				types[id] = otypes[id]
				g.Nodes = append(g.Nodes, NodeGene{id, otypes[id]})
			}
		}
		g.Conns = append(g.Conns, oc)
		have[oc.Innovation] = true
	}
	g.sorted = nil
}

func Compatibility(a, b *Genome) float64 {
	ac := a.sortedConns()
	bc := b.sortedConns()
	i, j := 0, 0
	matching, disjoint := 0, 0
	wdiff := 0.0
	for i < len(ac) && j < len(bc) {
		if ac[i].Innovation == bc[j].Innovation {
			matching++
			wdiff += math.Abs(ac[i].Weight - bc[j].Weight)
			i++
			j++
		} else if ac[i].Innovation < bc[j].Innovation {
			disjoint++
			i++
		} else {
			disjoint++
			j++
		}
	}
	disjoint += (len(ac) - i) + (len(bc) - j)
	n := math.Max(float64(len(ac)), float64(len(bc)))
	if n < 1 {
		n = 1
	}
	avgW := 0.0
	if matching > 0 {
		avgW = wdiff / float64(matching)
	}
	return float64(disjoint)/n + 0.4*avgW
}

// innWeight is what Compatibility reads of a connection.
type innWeight struct {
	Innovation int
	Weight     float64
}

// sortedConns returns the innovations and weights of the connections sorted
// by innovation. Only those two fields are kept: sorting the whole genes
// moved so much memory that it was half of a text-evolution generation once
// genomes had a few thousand connections. The result is cached: genomes
// stored in pools are not modified any more (children are copies), and
// Mutate drops the cache.
func (g *Genome) sortedConns() []innWeight {
	if g.sorted != nil && len(g.sorted) == len(g.Conns) {
		return g.sorted
	}
	c := make([]innWeight, len(g.Conns))
	// Innovations fit in 32 bits, so the sort runs on plain integers, the
	// innovation above the index, which is several times faster than a
	// sort with a comparison function; the order is the same.
	keys := make([]uint64, len(g.Conns))
	small := len(g.Conns) < 1<<32
	for i, x := range g.Conns {
		if x.Innovation < 0 || x.Innovation >= 1<<31 {
			small = false
			break
		}
		keys[i] = uint64(x.Innovation)<<32 | uint64(i)
	}
	if small {
		slices.Sort(keys)
		for i, k := range keys {
			x := g.Conns[k&(1<<32-1)]
			c[i] = innWeight{x.Innovation, x.Weight}
		}
	} else {
		for i, x := range g.Conns {
			c[i] = innWeight{x.Innovation, x.Weight}
		}
		slices.SortFunc(c, func(a, b innWeight) int { return cmp.Compare(a.Innovation, b.Innovation) })
	}
	g.sorted = c
	return c
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
