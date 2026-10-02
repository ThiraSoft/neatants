package main

import (
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
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
	out := sample(g, d, []int32{1, 2}, 30, 1, rand.New(rand.NewSource(1)))
	if len(out) != 30 {
		t.Fatal(len(out))
	}
	for _, id := range out {
		if id < 0 || int(id) >= d.Vocab() {
			t.Fatal(id)
		}
	}
}
