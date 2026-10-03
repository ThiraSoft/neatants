package main

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
)

// fakeEnc is a tokenizer reduced to a lookup table.
type fakeEnc map[string][]int32

func (f fakeEnc) Encode(text string, addBOS, parseSpecial bool) []int32 { return f[text] }

func (f fakeEnc) Decode(ids []int32, special bool) string {
	for k, v := range f {
		if slices.Equal(v, ids) {
			return k
		}
	}
	return "?"
}

func TestEncodePromptRejectsUnknownTokens(t *testing.T) {
	d := model.Synthetic(10, 8, 50, 1) // QwenID[i] = i
	enc := fakeEnc{"ROMEO": {3}, "zebra": {500}}
	if ids, err := encodePrompt(enc, d, "ROMEO"); err != nil || len(ids) != 1 || ids[0] != 3 {
		t.Fatal(ids, err)
	}
	_, err := encodePrompt(enc, d, "zebra")
	if err == nil || !strings.Contains(err.Error(), "zebra") {
		t.Fatalf("err %v", err)
	}
}

func TestSampleLength(t *testing.T) {
	d := model.Synthetic(10, 8, 50, 1)
	g := model.Grown(1, 8, 50)
	out := sample(g, d, nil, []int32{1, 2}, 30, 1, rand.New(rand.NewSource(1)))
	if len(out) != 30 {
		t.Fatal(len(out))
	}
	for _, id := range out {
		if id < 0 || int(id) >= d.Vocab() {
			t.Fatal(id)
		}
	}
}

// A champion with state banks samples with them: the same draws as a loop
// over model.Net, which is what the evaluation scored.
func TestSampleWithState(t *testing.T) {
	d := model.Synthetic(10, 8, 50, 1)
	g := model.GrownShape(1, model.Shape{Dim: 8, Banks: 2}, 50)
	got := sample(g, d, nil, []int32{1, 2}, 30, 1, rand.New(rand.NewSource(1)))
	rng := rand.New(rand.NewSource(1))
	net := model.NewNet(g, 8)
	o := make([]float32, 8)
	read := func(id int32) {
		for j, v := range net.Step(d.Row(id)) {
			o[j] = 2*v - 1
		}
	}
	read(1)
	read(2)
	for i := range 30 {
		id := draw(logits(d, o, model.LogitScale(g, 8)), 1, rng)
		if got[i] != id {
			t.Fatalf("token %d: sampled %d, want %d", i, got[i], id)
		}
		read(id)
	}
}

// The sampler scores tokens as the evaluation does: with o = 0 its logits are
// the unigram prior.
func TestLogitsStartFromThePrior(t *testing.T) {
	d := model.Synthetic(10, 8, 50, 1)
	got := logits(d, make([]float32, 8), 3)
	for j, p := range d.Prior() {
		if got[j] != float64(p) {
			t.Fatalf("token %d: logit %g, prior %g", j, got[j], p)
		}
	}
}

// Down the tree at temperature 1, tokens come out as often as the scoring
// says they should.
func TestDescendFollowsTreeBits(t *testing.T) {
	d := model.Synthetic(13, 8, 500, 3)
	model.Skew(d, 3)
	tr := model.BuildTree(d)
	o := []float32{0.3, -0.8, 0.5, 0.1}
	rng := rand.New(rand.NewSource(4))
	const n = 200000
	seen := make([]int, d.Vocab())
	for range n {
		seen[descend(tr, o, 2, 1, rng)]++
	}
	for id, c := range seen {
		p := math.Exp2(-ref.TreeBits(tr, o, 2, int32(id)))
		if got := float64(c) / n; math.Abs(got-p) > 4*math.Sqrt(p*(1-p)/n)+1e-4 {
			t.Fatalf("token %d drawn %.4f of the time, scored %.4f", id, got, p)
		}
	}
}
