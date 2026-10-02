package prep

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// byteEnc is a stand-in tokenizer: one token per byte, id = byte value.
type byteEnc struct{}

func (byteEnc) Encode(s string, _, _ bool) []int32 {
	ids := make([]int32, len(s))
	for i := range len(s) {
		ids[i] = int32(s[i])
	}
	return ids
}
func (byteEnc) Decode(ids []int32, _ bool) string {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = byte(id)
	}
	return string(b)
}

// fakeEmbd is a BF16 table of 256 rows of width 48, shaped like a GGUF
// token_embd: Shape[0] is the width, Shape[1] the number of rows.
func fakeEmbd(width int) tensors.Tensor {
	rng := rand.New(rand.NewSource(1))
	raw := make([]byte, 256*width*2)
	for i := 0; i < 256*width; i++ {
		bits := math.Float32bits(float32(rng.NormFloat64()))
		raw[2*i], raw[2*i+1] = byte(bits>>16), byte(bits>>24)
	}
	return tensors.Tensor{Name: "token_embd.weight", DType: "BF16", Shape: []int{width, 256}, Raw: raw}
}

const text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\nResolved. resolved.\nFirst Citizen:\nFirst, you know Caius Marcius is chief enemy.\nWe know it, we know it.\nLet us kill him.\n"

func TestSplitCutsAtNewline(t *testing.T) {
	train, val := Split([]byte(text))
	if string(train)+string(val) != text {
		t.Fatal("split loses bytes")
	}
	if train[len(train)-1] != '\n' {
		t.Fatalf("train ends with %q", train[len(train)-1])
	}
	if f := float64(len(train)) / float64(len(text)); f < 0.8 || f > 0.95 {
		t.Fatalf("train is %.2f of the text", f)
	}
}

func TestBuildRoundTripAndStats(t *testing.T) {
	train, val := Split([]byte(text))
	d, err := Build(train, val, byteEnc{}, fakeEmbd(48), 8)
	if err != nil {
		t.Fatal(err)
	}
	dec := func(ids []int32) string {
		q := make([]int32, len(ids))
		for i, id := range ids {
			q[i] = d.QwenID[id]
		}
		return byteEnc{}.Decode(q, false)
	}
	if dec(d.Train) != string(train) || dec(d.Val) != string(val) {
		t.Fatal("decoding the active ids does not give the text back")
	}
	n := 0
	for _, id := range d.Train {
		n += int(d.Bytes[id])
	}
	if n != len(train) {
		t.Fatalf("token bytes sum to %d, train has %d", n, len(train))
	}
	if d.Vocab() != len(strings.Split(uniq(text), "")) {
		t.Fatalf("vocab %d", d.Vocab())
	}
	for j := 0; j < d.Dim; j++ {
		var m, v float64
		for i := 0; i < d.Vocab(); i++ {
			m += float64(d.E[i*d.Dim+j])
		}
		m /= float64(d.Vocab())
		for i := 0; i < d.Vocab(); i++ {
			x := float64(d.E[i*d.Dim+j]) - m
			v += x * x
		}
		v /= float64(d.Vocab() - 1)
		if math.Abs(m) > 1e-4 || math.Abs(v-1) > 1e-3 {
			t.Fatalf("dim %d: mean %g var %g", j, m, v)
		}
	}
}

func uniq(s string) string {
	seen := map[rune]bool{}
	var b strings.Builder
	for _, r := range s {
		if !seen[r] {
			seen[r] = true
			b.WriteRune(r)
		}
	}
	return b.String()
}

func TestWriteLoad(t *testing.T) {
	train, val := Split([]byte(text))
	d, err := Build(train, val, byteEnc{}, fakeEmbd(48), 8)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "prep.bin")
	if err := d.Write(p); err != nil {
		t.Fatal(err)
	}
	e, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if Checksum(d) != Checksum(e) || len(e.Train) != len(d.Train) || len(e.Val) != len(d.Val) {
		t.Fatal("load differs from write")
	}
	os.WriteFile(p, []byte("garbage"), 0o644)
	if _, err := Load(p); err == nil {
		t.Fatal("garbage loaded")
	}
}

func TestQwenShakespeare(t *testing.T) {
	const gguf = "/mnt/data/LLMs_models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf"
	raw, err := os.ReadFile("../../../data/tinyshakespeare.txt")
	if err != nil {
		t.Skip("no corpus: make data")
	}
	g, err := tensors.OpenGGUF(gguf)
	if err != nil {
		t.Skip("no GGUF")
	}
	defer g.Close()
	voc, err := bytebpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	embd, err := g.Get("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	train, val := Split(raw)
	d, err := Build(train, val, voc, embd, 128)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("vocab %d, train %d tokens, val %d tokens", d.Vocab(), len(d.Train), len(d.Val))
	if d.Vocab() < 2000 || d.Vocab() > 40000 {
		t.Fatalf("vocab %d", d.Vocab())
	}
}

func TestPriorIsAddOneUnigram(t *testing.T) {
	d := &Data{QwenID: make([]int32, 5), Train: []int32{0, 0, 0, 2, 4, 4}}
	got := d.Prior()
	for j, c := range []int{3, 0, 1, 0, 2} {
		want := math.Log(float64(c+1) / float64(6+5))
		if math.Abs(float64(got[j])-want) > 1e-6 {
			t.Fatalf("token %d: prior %g, want %g", j, got[j], want)
		}
	}
	if &d.Prior()[0] != &got[0] {
		t.Fatal("the prior is computed again on every call")
	}
}
