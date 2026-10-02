// Package prep turns a text corpus into what the text evolution consumes: the
// token ids of the active vocabulary and a small whitened embedding table.
package prep

import (
	"bytes"
	"fmt"
	"math"
	"slices"

	"github.com/ThiraSoft/golem/tensors"
)

// Data is the prepared corpus. Ids in Train and Val are active ids, that is
// indexes into QwenID, Bytes and the rows of E.
type Data struct {
	Dim    int
	QwenID []int32   // per active token: its id in the Qwen vocabulary
	Bytes  []int32   // per active token: UTF-8 byte length of its text
	E      []float32 // Vocab()*Dim, row-major, whitened
	Train  []int32   // active ids
	Val    []int32
}

// Encoder is the part of a tokenizer that prep needs.
type Encoder interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Decode(ids []int32, special bool) string
}

// Vocab is the number of active tokens.
func (d *Data) Vocab() int { return len(d.QwenID) }

// Row is the embedding of an active token, a view into E.
func (d *Data) Row(id int32) []float32 {
	return d.E[int(id)*d.Dim : (int(id)+1)*d.Dim]
}

// Split cuts the text about 90/10 into train and validation. The cut moves
// forward to just after a newline so that no line is shared by both sides.
func Split(text []byte) (train, val []byte) {
	cut := len(text) * 9 / 10
	if i := bytes.IndexByte(text[cut:], '\n'); i >= 0 {
		cut += i + 1
	}
	return text[:cut], text[cut:]
}

// Build tokenizes both parts, keeps the tokens that occur in either, and
// reduces their Qwen embeddings to dim whitened dimensions.
func Build(train, val []byte, enc Encoder, embd tensors.Tensor, dim int) (*Data, error) {
	tr := enc.Encode(string(train), false, false)
	va := enc.Encode(string(val), false, false)

	seen := map[int32]bool{}
	for _, id := range tr {
		seen[id] = true
	}
	for _, id := range va {
		seen[id] = true
	}
	active := make([]int32, 0, len(seen))
	for id := range seen {
		active = append(active, id)
	}
	slices.Sort(active)
	remap := make(map[int32]int32, len(active))
	for i, q := range active {
		remap[q] = int32(i)
	}
	n := len(active)

	if embd.DType != "BF16" || len(embd.Shape) != 2 {
		return nil, fmt.Errorf("prep: embedding table must be a 2D BF16 tensor, got %s %v", embd.DType, embd.Shape)
	}
	width, rows := embd.Shape[0], embd.Shape[1]
	if len(embd.Raw) < rows*width*2 {
		return nil, fmt.Errorf("prep: embedding table holds %d bytes, %dx%d needs %d", len(embd.Raw), rows, width, rows*width*2)
	}
	if dim < 1 || dim > width || dim > n-1 {
		return nil, fmt.Errorf("prep: dim %d does not fit the embedding width %d and %d active tokens", dim, width, n)
	}

	d := &Data{Dim: dim, QwenID: active, Bytes: make([]int32, n)}
	x := make([]float64, n*width)
	for i, q := range active {
		if int(q) >= rows {
			return nil, fmt.Errorf("prep: qwen id %d is outside the %d rows of the embedding table", q, rows)
		}
		d.Bytes[i] = int32(len(enc.Decode([]int32{q}, false)))
		for j := range width {
			k := int(q)*width + j
			bits := uint32(embd.Raw[2*k]) | uint32(embd.Raw[2*k+1])<<8
			x[i*width+j] = float64(math.Float32frombits(bits << 16))
		}
	}
	d.E = pca(x, n, width, dim)
	d.Train = remapAll(tr, remap)
	d.Val = remapAll(va, remap)
	return d, nil
}

func remapAll(ids []int32, remap map[int32]int32) []int32 {
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = remap[id]
	}
	return out
}
