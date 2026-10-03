package model

import (
	"math"
	"math/bits"
	"sort"
	"sync"

	"github.com/ThiraSoft/neatants/internal/textevo/prep"
)

// PathEnd fills the levels of a path below its leaf.
const PathEnd = math.MaxUint32

// PathBit marks a step of a path that goes to the second child.
const PathBit = 1 << 31

// Tree is a balanced binary tree whose leaves are the active tokens, for the
// tree scoring: a network has one output a level, output k decides the branch
// taken at level k, and a token's probability is the product of the decisions
// on its path. Siblings are close in embedding space, so that a decision
// means something.
type Tree struct {
	// Depth is the length of the longest path, ceil(log2 Vocab).
	Depth int
	// Bias is the log-odds of the second child of each internal node under
	// the add-one unigram of the train tokens, ln(S1/S0), so that a network
	// whose outputs are zero is exactly that unigram.
	Bias []float32
	// Path holds Depth words a token: node | PathBit for the second child,
	// for each level of its path from the root, then PathEnd.
	Path []uint32
}

// TokenPath is the path of token id, Depth words.
func (t *Tree) TokenPath(id int32) []uint32 {
	return t.Path[int(id)*t.Depth : (int(id)+1)*t.Depth]
}

// treeSplitSerial is the subtree size below which BuildTree stops starting
// goroutines.
const treeSplitSerial = 512

// BuildTree splits the active tokens of d in two at the median of their
// projection on the principal axis of their embeddings, and again in each
// half, down to single tokens. The nodes are numbered in preorder. The tree
// depends only on d, so a run and its samples rebuild the same one.
func BuildTree(d *prep.Data) *Tree {
	v := d.Vocab()
	if v < 2 {
		panic("textevo: the tree scoring needs at least two tokens")
	}
	t := &Tree{Depth: bits.Len(uint(v - 1)), Bias: make([]float32, v-1)}
	t.Path = make([]uint32, v*t.Depth)
	for i := range t.Path {
		t.Path[i] = PathEnd
	}
	count := make([]float64, v)
	for i := range count {
		count[i] = 1
	}
	for _, id := range d.Train {
		count[id]++
	}
	set := make([]int32, v)
	for i := range set {
		set[i] = int32(i)
	}
	var wg sync.WaitGroup
	var split func(set []int32, node uint32, level int)
	split = func(set []int32, node uint32, level int) {
		sortByAxis(d, set)
		lo, hi := set[:len(set)/2], set[len(set)/2:]
		s0, s1 := 0.0, 0.0
		for _, id := range lo {
			s0 += count[id]
			t.Path[int(id)*t.Depth+level] = node
		}
		for _, id := range hi {
			s1 += count[id]
			t.Path[int(id)*t.Depth+level] = node | PathBit
		}
		t.Bias[node] = float32(math.Log(s1 / s0))
		// A subtree of n leaves has n-1 internal nodes.
		for _, c := range []struct {
			set  []int32
			node uint32
		}{{lo, node + 1}, {hi, node + uint32(len(lo))}} {
			if len(c.set) < 2 {
				continue
			}
			if len(c.set) < treeSplitSerial {
				split(c.set, c.node, level+1)
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				split(c.set, c.node, level+1)
			}()
		}
	}
	split(set, 0, 0)
	wg.Wait()
	return t
}

// sortByAxis orders set by the projection of its embeddings on their
// principal axis, found by power iteration from the first token's offset to
// the mean; ties keep the id order.
func sortByAxis(d *prep.Data, set []int32) {
	dim := d.Dim
	mean := make([]float64, dim)
	for _, id := range set {
		for k, x := range d.Row(id) {
			mean[k] += float64(x)
		}
	}
	for k := range mean {
		mean[k] /= float64(len(set))
	}
	axis := make([]float64, dim)
	for k, x := range d.Row(set[0]) {
		axis[k] = float64(x) - mean[k] + 1e-3
	}
	next := make([]float64, dim)
	for range 30 {
		norm := 0.0
		for _, x := range axis {
			norm += x * x
		}
		if norm == 0 {
			break
		}
		norm = math.Sqrt(norm)
		for k := range axis {
			axis[k] /= norm
		}
		clear(next)
		for _, id := range set {
			row := d.Row(id)
			p := 0.0
			for k, x := range row {
				p += (float64(x) - mean[k]) * axis[k]
			}
			for k, x := range row {
				next[k] += p * (float64(x) - mean[k])
			}
		}
		axis, next = next, axis
	}
	proj := make(map[int32]float64, len(set))
	for _, id := range set {
		p := 0.0
		for k, x := range d.Row(id) {
			p += float64(x) * axis[k]
		}
		proj[id] = p
	}
	sort.Slice(set, func(a, b int) bool {
		pa, pb := proj[set[a]], proj[set[b]]
		if pa != pb {
			return pa < pb
		}
		return set[a] < set[b]
	})
}

// Inputs of a network under the tree scoring: what row a token gives it.
const (
	InputEmb  = "emb"  // the embedding, as under the softmax
	InputCode = "code" // the path of the token, one value a level
	InputBoth = "both" // the embedding then the path
)

// InputData returns d with the rows a network reads under input mode in. The
// path of a token reads +1 for the second child, -1 for the first and 0
// below its leaf, a binary embedding of Depth values in which tokens close in
// the tree share their first values. Only Dim and E differ from d.
func InputData(d *prep.Data, t *Tree, in string) *prep.Data {
	if in == InputEmb {
		return d
	}
	w := t.Depth
	if in == InputBoth {
		w += d.Dim
	}
	e := make([]float32, 0, d.Vocab()*w)
	for id := range int32(d.Vocab()) {
		if in == InputBoth {
			e = append(e, d.Row(id)...)
		}
		for _, s := range t.TokenPath(id) {
			switch {
			case s == PathEnd:
				e = append(e, 0)
			case s&PathBit != 0:
				e = append(e, 1)
			default:
				e = append(e, -1)
			}
		}
	}
	return &prep.Data{Dim: w, QwenID: d.QwenID, Bytes: d.Bytes, E: e, Train: d.Train, Val: d.Val}
}
