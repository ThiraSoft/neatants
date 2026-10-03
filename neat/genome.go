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
	nodeWiden         // hidden node of a block added by a widening
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
			// In input order, so that the genes of a fresh genome are in the
			// order of their innovations (numbered the first time they are
			// met) and the sorted cache of a dense genome is almost free.
			reads = rand.Perm(inputs)[:min(MinimalLinks, inputs)]
			slices.Sort(reads)
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
	if rand.Float64() < WidenRate {
		g.widenMutation()
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

// Widening, per Mutate: a block of WidenNodes hidden neurons, each reading
// WidenIn random inputs and read by WidenOut random neurons. Single-node
// mutations grow networks deep and thin, one input and one output a neuron;
// a learner that tunes many weights at once (textevo -learn) needs width. Off
// unless WidenRate is set.
var (
	WidenRate  = 0.0
	WidenNodes = 8
	WidenIn    = 16
	WidenOut   = 8
)

// Widen adds a block of hidden neurons as the widening mutation does.
func (g *Genome) Widen() {
	g.sorted = nil
	g.widenMutation()
}

// widenMutation adds the block with its outgoing weights at zero, so that the
// network computes exactly what it did (Net2Net): only a learner, or later
// mutations, make the new neurons matter. The incoming weights are drawn with
// a spread of 1/sqrt(WidenIn) so the new sums start in the useful range.
func (g *Genome) widenMutation() {
	var srcs, dsts []int
	// The block reads only inputs and the bias, which depend on nothing: a
	// source computed by the network could change the order in which
	// BuildNetwork breaks cycles, and with it which links read the previous
	// tick, so the network would no longer compute the same thing.
	for _, n := range g.Nodes {
		if n.Type == Sensor || n.Type == Bias {
			srcs = append(srcs, n.ID)
		}
		if n.Type == Output || n.Type == Hidden || n.Type == Memory {
			dsts = append(dsts, n.ID)
		}
	}
	if len(srcs) == 0 || len(dsts) == 0 || WidenNodes < 1 {
		return
	}
	pick := func(from []int, k int) []int {
		k = min(k, len(from))
		out := make([]int, 0, k)
		for _, i := range rand.Perm(len(from))[:k] {
			out = append(out, from[i])
		}
		return out
	}
	in, out := pick(srcs, WidenIn), pick(dsts, WidenOut)
	spread := 1 / math.Sqrt(float64(len(in)))
	for j := range WidenNodes {
		nid := g.globalNodeID([3]int{nodeWiden, out[0], j})
		g.Nodes = append(g.Nodes, NodeGene{nid, Hidden})
		for _, a := range in {
			g.Conns = append(g.Conns, ConnGene{In: a, Out: nid, Weight: rand.NormFloat64() * spread, Enabled: true, Innovation: getInnovation(a, nid)})
		}
		for _, b := range out {
			g.Conns = append(g.Conns, ConnGene{In: nid, Out: b, Weight: 0, Enabled: true, Innovation: getInnovation(nid, b)})
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
	// Which nodes are read is a table by node ID when the IDs are small
	// enough, since a map write for each of the sixteen thousand genes of a
	// dense genome was a fifth of a text-evolution generation.
	lo, top := 0, 0
	for _, n := range g.Nodes {
		lo, top = min(lo, n.ID), max(top, n.ID)
	}
	var table []bool
	var set map[int]bool
	if lo >= 0 && top < 64*len(g.Nodes)+1024 {
		table = make([]bool, top+1)
	} else {
		set = make(map[int]bool, len(g.Nodes))
	}
	for _, c := range g.Conns {
		if c.Enabled && c.In != c.Out {
			if c.In >= 0 && c.In < len(table) {
				table[c.In] = true
			} else if set != nil {
				set[c.In] = true
			}
		}
	}
	read := func(id int) bool {
		if table != nil {
			return id >= 0 && id < len(table) && table[id]
		}
		return set[id]
	}
	dead := map[int]bool{}
	nodes := g.Nodes[:0]
	for _, n := range g.Nodes {
		if (n.Type == Hidden || n.Type == Memory) && !read(n.ID) && rand.Float64() < PruneNodeRate {
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
func (g *Genome) AddMemory() {
	g.sorted = nil
	g.addMemoryMutation()
}

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
	// A rate of zero or less perturbs nothing, and it must return before the
	// geometric law below, whose logarithm of 1-rate is zero or undefined.
	if rate <= 0 {
		return
	}
	// Each connection is perturbed with probability rate. Rather than one
	// draw per connection, the gap to the next perturbed one is drawn from
	// the geometric law of that test, which is the same choice in a few draws
	// instead of sixteen thousand for a dense genome.
	logq := math.Log1p(-rate)
	gap := func() int {
		if rate >= 1 {
			return 0
		}
		return int(math.Min(math.Log(1-rand.Float64())/logq, float64(len(g.Conns))))
	}
	for i := gap(); i < len(g.Conns); i += 1 + gap() {
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
	// If better holds the same innovation twice, only the first copy can take
	// the other parent's gene. That is rare and harmless.
	if better.MutPower > 0 && other.MutPower > 0 {
		child.MutPower = math.Sqrt(better.MutPower * other.MutPower)
	}
	if len(child.Traits) == len(other.Traits) {
		for i := range child.Traits {
			t := rand.Float64()
			child.Traits[i] = child.Traits[i]*t + other.Traits[i]*(1-t)
		}
	}
	// The genes are paired by walking both parents' innovation-sorted
	// caches side by side: a map of the other parent's genes cost more than
	// the rest of a text-evolution generation once genomes had sixteen
	// thousand connections.
	bs, os := better.indexedConns(), other.indexedConns()
	var only []int // where in other.Conns the genes better lacks sit
	i := 0
	for _, o := range os {
		for i < len(bs) && bs[i].Innovation < o.Innovation {
			i++
		}
		switch {
		case i < len(bs) && bs[i].Innovation == o.Innovation:
			// A matching gene comes whole from one parent, weight and
			// state: the classic "disabled in either parent → 75 %
			// disabled" rule ratchets connections off generation after
			// generation.
			if rand.Float64() < 0.5 {
				oc := other.Conns[o.Index]
				child.Conns[bs[i].Index].Weight = oc.Weight
				child.Conns[bs[i].Index].Enabled = oc.Enabled
			}
			i++
		case i > 0 && bs[i-1].Innovation == o.Innovation:
			// A second gene of other with an innovation better has.
		default:
			only = append(only, o.Index)
		}
	}
	slices.Sort(only)
	child.importGenes(other, only)
	return child
}

// ImportRate is the chance that a gene only the weaker parent carries is
// passed on too, so a child inherits structure from both lineages.
var ImportRate = 0.25

// importGenes copies some of other's enabled disjoint and excess connections,
// the ones at the positions only lists, adding the nodes they need. Node IDs
// are global, but genomes saved before that had local ones: a gene whose node
// ID exists here with another type belongs to a different neuron and is
// skipped.
func (g *Genome) importGenes(other *Genome, only []int) {
	have := map[int]bool{}
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
	for _, k := range only {
		oc := other.Conns[k]
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

// PrepareCompatibility fills the cache Compatibility and Crossover read, so
// that several goroutines may then compare or cross g at once: the cache is
// otherwise filled on first use, which would be a race between them.
func (g *Genome) PrepareCompatibility() { g.indexedConns() }

// indexedConns is sortedConns for a reader that follows Index into Conns. A
// genome whose genes were reordered in place without Mutate would keep a
// stale cache of the same length, which only blurs a distance but would pair
// the wrong genes, so a few entries spread over the cache are checked against
// Conns and the cache is rebuilt if one is off. All of them would cost a
// fifth of a dense generation; neat itself always drops the cache when it
// changes the genes.
func (g *Genome) indexedConns() []innWeight {
	s := g.sortedConns()
	for k := 0; k < len(s); k += max(1, len(s)/16) {
		if e := s[k]; g.Conns[e.Index].Innovation != e.Innovation {
			g.sorted = nil
			return g.sortedConns()
		}
	}
	return s
}

// innWeight is what Compatibility reads of a connection, and where it sits in
// Conns, which Crossover follows to the whole gene.
type innWeight struct {
	Innovation int
	Weight     float64
	Index      int
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
	small := uint64(len(g.Conns)) < 1<<32
	for i, x := range g.Conns {
		if x.Innovation < 0 || int64(x.Innovation) >= 1<<31 {
			small = false
			break
		}
		keys[i] = uint64(x.Innovation)<<32 | uint64(i)
	}
	if small {
		sortMostlySorted(keys)
		for i, k := range keys {
			x := g.Conns[k&(1<<32-1)]
			c[i] = innWeight{x.Innovation, x.Weight, int(k & (1<<32 - 1))}
		}
	} else {
		for i, x := range g.Conns {
			c[i] = innWeight{x.Innovation, x.Weight, i}
		}
		slices.SortStableFunc(c, func(a, b innWeight) int { return cmp.Compare(a.Innovation, b.Innovation) })
	}
	g.sorted = c
	return c
}

// sortMostlySorted sorts keys that are usually in order up to a short tail:
// genes are appended as they appear, and most get a new, higher innovation.
// Only the tail is sorted, then merged with the ordered head, so a dense
// genome of sixteen thousand genes is not sorted from scratch for each child.
func sortMostlySorted(keys []uint64) {
	p := 1
	for p < len(keys) && keys[p-1] <= keys[p] {
		p++
	}
	if p >= len(keys) {
		return
	}
	tail := keys[p:]
	slices.Sort(tail)
	if keys[p-1] <= tail[0] {
		return
	}
	head := slices.Clone(keys[:p])
	i, j, k := 0, 0, 0
	for i < len(head) && j < len(tail) {
		// tail[j] is never overwritten before it is read: k = i+j < p+j.
		if head[i] <= tail[j] {
			keys[k] = head[i]
			i++
		} else {
			keys[k] = tail[j]
			j++
		}
		k++
	}
	for ; i < len(head); i, k = i+1, k+1 {
		keys[k] = head[i]
	}
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
