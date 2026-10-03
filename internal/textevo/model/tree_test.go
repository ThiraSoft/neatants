package model

import (
	"math"
	"math/bits"
	"testing"
)

// Every token is a leaf: paths start at the root, stay inside the tree, and
// no two tokens share one.
func TestTreePaths(t *testing.T) {
	for _, v := range []int{2, 3, 50, 64, 300} {
		d := Synthetic(v, 8, 1000, int64(v))
		tr := BuildTree(d)
		if tr.Depth != bits.Len(uint(v-1)) || len(tr.Bias) != v-1 {
			t.Fatalf("vocab %d: depth %d, %d nodes", v, tr.Depth, len(tr.Bias))
		}
		seen := map[string]int32{}
		for id := range int32(v) {
			p := tr.TokenPath(id)
			if p[0]&^PathBit != 0 {
				t.Fatalf("vocab %d: token %d does not start at the root", v, id)
			}
			key := ""
			for _, w := range p {
				if w == PathEnd {
					break
				}
				if int(w&^PathBit) >= v-1 {
					t.Fatalf("vocab %d: token %d goes through node %d", v, id, w&^PathBit)
				}
				key += string(rune(w))
			}
			if o, ok := seen[key]; ok {
				t.Fatalf("vocab %d: tokens %d and %d share a path", v, o, id)
			}
			seen[key] = id
		}
	}
}

// With every output at zero, a token's probability is the product of the
// sigmoids of the biases on its path, which must be its add-one unigram.
func TestTreeZeroOutputIsUnigram(t *testing.T) {
	d := Synthetic(300, 16, 3000, 5)
	Skew(d, 5)
	tr := BuildTree(d)
	n := float64(len(d.Train) + d.Vocab())
	count := make([]float64, d.Vocab())
	for _, id := range d.Train {
		count[id]++
	}
	sum := 0.0
	for id := range int32(d.Vocab()) {
		lp := 0.0
		for _, w := range tr.TokenPath(id) {
			if w == PathEnd {
				break
			}
			z := float64(tr.Bias[w&^PathBit])
			if w&PathBit == 0 {
				z = -z
			}
			lp -= math.Log1p(math.Exp(-z))
		}
		want := math.Log((count[id] + 1) / n)
		if math.Abs(lp-want) > 1e-5 {
			t.Fatalf("token %d: ln p %g, unigram %g", id, lp, want)
		}
		sum += math.Exp(lp)
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("probabilities sum to %g", sum)
	}
}

// The tree depends only on the data, whatever the goroutines did.
func TestTreeIsDeterministic(t *testing.T) {
	d := Synthetic(3000, 16, 1000, 9)
	a, b := BuildTree(d), BuildTree(d)
	for i := range a.Path {
		if a.Path[i] != b.Path[i] {
			t.Fatal("two builds differ")
		}
	}
}

func TestInputData(t *testing.T) {
	d := Synthetic(50, 8, 100, 3)
	tr := BuildTree(d)
	if InputData(d, tr, InputEmb) != d {
		t.Fatal("emb input is not the data itself")
	}
	c, b := InputData(d, tr, InputCode), InputData(d, tr, InputBoth)
	if c.Dim != tr.Depth || b.Dim != 8+tr.Depth {
		t.Fatalf("dims %d and %d", c.Dim, b.Dim)
	}
	for id := range int32(50) {
		for k, s := range tr.TokenPath(id) {
			want := float32(-1)
			if s == PathEnd {
				want = 0
			} else if s&PathBit != 0 {
				want = 1
			}
			if c.Row(id)[k] != want || b.Row(id)[8+k] != want {
				t.Fatalf("token %d level %d: %g and %g, want %g", id, k, c.Row(id)[k], b.Row(id)[8+k], want)
			}
		}
		for k := range 8 {
			if b.Row(id)[k] != d.Row(id)[k] {
				t.Fatalf("token %d: both does not start with the embedding", id)
			}
		}
	}
}
