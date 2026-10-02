package neat

import (
	"math/rand"
	"slices"
	"testing"
)

// The cache Compatibility and Crossover read lists the connections by
// innovation, ties by position, each with its own weight and position.
func TestSortedConnsOrder(t *testing.T) {
	g := NewGenome(1, 8, 8)
	// A sorted head, an unsorted tail and a repeated innovation, as genomes
	// get them from mutations that reuse an older innovation.
	for i := range 40 {
		g.Conns = append(g.Conns, ConnGene{Innovation: 1 + rand.Intn(60), Weight: float64(i)})
	}
	for _, shuffle := range []bool{false, true} {
		if shuffle {
			rand.Shuffle(len(g.Conns), func(i, j int) { g.Conns[i], g.Conns[j] = g.Conns[j], g.Conns[i] })
		}
		g.sorted = nil
		s := g.sortedConns()
		if len(s) != len(g.Conns) {
			t.Fatalf("%d entries for %d connections", len(s), len(g.Conns))
		}
		seen := make([]bool, len(g.Conns))
		for k, e := range s {
			c := g.Conns[e.Index]
			if seen[e.Index] || c.Innovation != e.Innovation || c.Weight != e.Weight {
				t.Fatalf("entry %d %+v does not match connection %d %+v", k, e, e.Index, c)
			}
			seen[e.Index] = true
			if k > 0 && (s[k-1].Innovation > e.Innovation || s[k-1].Innovation == e.Innovation && s[k-1].Index > e.Index) {
				t.Fatalf("entries %d and %d out of order: %+v %+v", k-1, k, s[k-1], e)
			}
		}
	}
}

// relatives returns two genomes of one lineage that went separate ways, with
// enough structural mutations to have disjoint genes and new nodes.
func relatives() (*Genome, *Genome) {
	defer func(k int, a, m float64) { MinimalLinks, AddNodeRate, AddMemoryRate = k, a, m }(MinimalLinks, AddNodeRate, AddMemoryRate)
	MinimalLinks, AddNodeRate, AddMemoryRate = 4, 0.3, 0.1
	base := NewGenome(1, 8, 8)
	a, b := base.Copy(), base.Copy()
	for range 30 {
		a.Mutate()
		b.Mutate()
	}
	return a, b
}

// A child keeps every gene of the better parent in its order, takes each
// matching gene whole from one parent or the other, and with ImportRate 1
// every enabled gene only the other parent has, when its nodes fit.
func TestCrossoverGenes(t *testing.T) {
	defer func(r float64) { ImportRate = r }(ImportRate)
	fromOther, matching := 0, 0
	for trial := range 50 {
		a, b := relatives()
		ImportRate = float64(trial % 2)
		c := Crossover(a, b, 99)
		if c.ID != 99 {
			t.Fatal(c.ID)
		}
		bi := map[int]ConnGene{}
		for _, x := range b.Conns {
			bi[x.Innovation] = x
		}
		ai := map[int]bool{}
		for k, x := range a.Conns {
			ai[x.Innovation] = true
			y := c.Conns[k]
			if y.Innovation != x.Innovation || y.In != x.In || y.Out != x.Out {
				t.Fatalf("gene %d of the better parent changed: %+v -> %+v", k, x, y)
			}
			if o, ok := bi[x.Innovation]; ok {
				if o.Weight != x.Weight || o.Enabled != x.Enabled {
					matching++
				}
				switch {
				case y.Weight == x.Weight && y.Enabled == x.Enabled:
				case y.Weight == o.Weight && y.Enabled == o.Enabled:
					if o.Weight != x.Weight || o.Enabled != x.Enabled {
						fromOther++
					}
				default:
					t.Fatalf("matching gene %+v comes from neither parent (%+v, %+v)", y, x, o)
				}
			} else if y.Weight != x.Weight || y.Enabled != x.Enabled {
				t.Fatalf("gene %+v only the better parent has changed to %+v", x, y)
			}
		}
		nodes := map[int]NodeType{}
		for _, n := range c.Nodes {
			nodes[n.ID] = n.Type
		}
		var imported []int
		for _, y := range c.Conns[len(a.Conns):] {
			if ai[y.Innovation] || !slices.Contains(b.Conns, y) {
				t.Fatalf("imported gene %+v is not a gene only the other parent has", y)
			}
			if _, ok := nodes[y.In]; !ok {
				t.Fatalf("imported gene %+v reads a missing node", y)
			}
			imported = append(imported, y.Innovation)
		}
		var want []int
		have := map[int]bool{}
		for _, n := range a.Nodes {
			have[n.ID] = true
		}
		types := map[int]NodeType{}
		for _, n := range b.Nodes {
			types[n.ID] = n.Type
		}
		for _, x := range b.Conns {
			_, okIn := types[x.In]
			_, okOut := types[x.Out]
			if ImportRate == 1 && x.Enabled && !ai[x.Innovation] && okIn && okOut && !slices.Contains(want, x.Innovation) {
				want = append(want, x.Innovation)
			}
		}
		if !slices.Equal(imported, want) {
			t.Fatalf("import rate %g: imported %v, want %v", ImportRate, imported, want)
		}
	}
	if matching == 0 || fromOther < matching/4 || fromOther > 3*matching/4 {
		t.Fatalf("%d of %d matching genes from the other parent, want about half", fromOther, matching)
	}
}

// sortMostlySorted must sort whatever the split between the ordered head and
// the tail, the head empty, the tail empty and repeated keys included.
func TestSortMostlySorted(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 2000 {
		n := rng.Intn(40)
		keys := make([]uint64, n)
		for i := range keys {
			keys[i] = uint64(rng.Intn(30))
		}
		head := rng.Intn(n + 1)
		slices.Sort(keys[:head])
		want := slices.Sorted(slices.Values(keys))
		sortMostlySorted(keys)
		if !slices.Equal(keys, want) {
			t.Fatalf("trial %d: %v, want %v", trial, keys, want)
		}
	}
}

// A weight mutation perturbs WeightsPerMutation weights on average, each
// connection with the same chance, whatever the size of the genome.
func TestWeightMutationCount(t *testing.T) {
	defer func(w float64) { WeightsPerMutation = w }(WeightsPerMutation)
	for _, w := range []float64{3, 30} {
		WeightsPerMutation = w
		g := NewGenome(1, 40, 40)
		for len(g.Conns) < 1000 {
			g.Conns = append(g.Conns, ConnGene{Innovation: len(g.Conns) + 1, Weight: 0.5})
		}
		const trials = 4000
		total, first, last := 0, 0, 0
		for range trials {
			for i := range g.Conns {
				g.Conns[i].Weight = 0.5
			}
			g.mutateWeightsAdaptive()
			for i, c := range g.Conns {
				if c.Weight != 0.5 {
					total++
					if i < 500 {
						first++
					} else {
						last++
					}
				}
			}
		}
		mean := float64(total) / trials
		t.Logf("WeightsPerMutation %g: %.3f perturbed on average, %d in the first half, %d in the second", w, mean, first, last)
		if mean < 0.95*w || mean > 1.05*w {
			t.Fatalf("WeightsPerMutation %g: %.3f perturbed on average", w, mean)
		}
		if d := float64(first-last) / float64(total); d < -0.05 || d > 0.05 {
			t.Fatalf("WeightsPerMutation %g: %d in the first half, %d in the second", w, first, last)
		}
	}
}

// A weight mutation rate of zero or less does nothing and does not panic.
func TestWeightMutationNoRate(t *testing.T) {
	defer func(w float64) { WeightsPerMutation = w }(WeightsPerMutation)
	for _, w := range []float64{0, -1} {
		WeightsPerMutation = w
		g := NewGenome(1, 4, 4)
		for len(g.Conns) < 20 {
			g.Conns = append(g.Conns, ConnGene{Innovation: len(g.Conns) + 1, Weight: 0.5})
		}
		for i := range g.Conns {
			g.Conns[i].Weight = 0.5
		}
		g.mutateWeightsAdaptive()
		for _, c := range g.Conns {
			if c.Weight != 0.5 {
				t.Fatalf("WeightsPerMutation %g changed a weight", w)
			}
		}
		g.Mutate()
	}
}
