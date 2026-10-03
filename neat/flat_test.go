package neat

import (
	"math/rand"
	"slices"
	"sort"
	"testing"
)

// TestFlatMatchesNetwork runs the level-order interpreter and the network
// side by side: same inputs, bit-identical outputs, cells and weights.
func TestFlatMatchesNetwork(t *testing.T) {
	for seed := int64(1); seed <= 5; seed++ {
		net := bigGenome(seed, 67, 8).BuildNetwork()
		f := net.Flat()
		s := f.NewState()
		r := rand.New(rand.NewSource(seed))
		for tick := range 200 {
			in, _ := randomInputs(r, 67)
			want := net.Activate(in)
			got := f.Activate(s, in)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("seed %d tick %d output %d: flat %v, network %v", seed, tick, i, got[i], want[i])
				}
			}
			for i := range net.cell {
				if s.Cell[i] != net.cell[i] {
					t.Fatalf("seed %d tick %d cell %d differs", seed, tick, i)
				}
			}
			for k := range net.weight {
				if s.Weight[k] != net.weight[k] {
					t.Fatalf("seed %d tick %d weight %d differs", seed, tick, k)
				}
			}
		}
	}
}

// TestFlatLevels checks the level invariant the GPU relies on: a forward
// edge always comes from a lower level, so a level only reads finished nodes.
func TestFlatLevels(t *testing.T) {
	f := bigGenome(2, 67, 8).BuildNetwork().Flat()
	n := int32(f.Nodes())
	level := make([]int32, n)
	for l := 0; l+1 < len(f.LevelStart); l++ {
		for _, ni := range f.Order[f.LevelStart[l]:f.LevelStart[l+1]] {
			level[ni] = int32(l + 1)
		}
	}
	for _, ni := range f.Order {
		for g := range int32(NumGates) {
			slot := ni*int32(NumGates) + g
			for k := f.Off[slot]; k < f.Off[slot+1]; k++ {
				if src := f.From[k]; src < n && level[src] >= level[ni] {
					t.Fatalf("node %d (level %d) reads node %d (level %d) of this tick", ni, level[ni], src, level[src])
				}
			}
		}
	}
	if len(f.LevelStart) < 3 {
		t.Fatalf("only %d levels: the test genome is too shallow", len(f.LevelStart)-1)
	}
}

// Flat orders the nodes by level, keeping the network's order inside a
// level, as a stable sort by level would.
func TestFlatOrderIsStableByLevel(t *testing.T) {
	for seed := int64(1); seed <= 5; seed++ {
		net := bigGenome(seed, 67, 8).BuildNetwork()
		f := net.Flat()
		level := map[int32]int{}
		for l := 0; l+1 < len(f.LevelStart); l++ {
			for _, ni := range f.Order[f.LevelStart[l]:f.LevelStart[l+1]] {
				level[ni] = l
			}
		}
		want := append([]int32(nil), net.order...)
		sort.SliceStable(want, func(a, b int) bool { return level[want[a]] < level[want[b]] })
		if !slices.Equal(want, f.Order) || len(level) != len(net.order) {
			t.Fatalf("seed %d: order %v, want %v", seed, f.Order, want)
		}
	}
}

func TestEdgeGenes(t *testing.T) {
	for seed := range int64(20) {
		rand.Seed(seed)
		g := NewGenome(1, 6, 4)
		for range 200 {
			g.Mutate()
		}
		g.AddMemory()
		g.AddMemory()
		if len(g.Conns) > 3 {
			g.Conns[2].Enabled = false
		}
		f := g.BuildNetwork().Flat()
		genes := g.EdgeGenes()
		if len(genes) != len(f.Weight) {
			t.Fatalf("seed %d: %d genes for %d edges", seed, len(genes), len(f.Weight))
		}
		for k, i := range genes {
			if float32(g.Conns[i].Weight) != f.Weight[k] || !g.Conns[i].Enabled {
				t.Fatalf("seed %d edge %d: gene %d weight %g, edge %g", seed, k, i, g.Conns[i].Weight, f.Weight[k])
			}
		}
	}
}

// A widening adds the block and changes nothing the network computes: the
// new neurons are read with weight zero.
func TestWidenKeepsTheFunction(t *testing.T) {
	for seed := range int64(10) {
		rand.Seed(seed)
		g := NewGenome(1, 12, 5)
		for range 150 {
			g.Mutate()
		}
		g.AddMemory()
		w := g.Copy()
		w.Widen()
		if len(w.Nodes) != len(g.Nodes)+WidenNodes {
			t.Fatalf("seed %d: %d nodes, want %d", seed, len(w.Nodes), len(g.Nodes)+WidenNodes)
		}
		fa, fb := g.BuildNetwork().Flat(), w.BuildNetwork().Flat()
		sa, sb := fa.NewState(), fb.NewState()
		in := make([]float32, 12)
		for tick := range 30 {
			for i := range in {
				in[i] = float32(rand.NormFloat64())
			}
			a, b := fa.Activate(sa, in), fb.Activate(sb, in)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("seed %d tick %d: output %d is %g, was %g", seed, tick, i, b[i], a[i])
				}
			}
		}
	}
}
