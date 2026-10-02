// Package baseline scores classic n-gram models over the active token ids, in
// the same bits per byte as the evolved networks, so the two can be compared.
package baseline

import (
	"math"

	"github.com/ThiraSoft/neatants/internal/textevo/prep"
)

// Result holds bits per byte on the validation tokens.
type Result struct{ Unigram, Bigram, Trigram float64 }

// counts holds the n-gram statistics of a token sequence. Ids are below 2^21,
// so a trigram fits in one uint64 key.
type counts struct {
	vocab int
	n     int
	c1    []int
	c2    map[uint64]int // (a,b)
	h1    map[uint64]int // a as a bigram context
	c3    map[uint64]int // (a,b,c)
	h2    map[uint64]int // (a,b) as a trigram context
}

func key2(a, b int32) uint64    { return uint64(a)<<32 | uint64(b) }
func key3(a, b, c int32) uint64 { return uint64(a)<<42 | uint64(b)<<21 | uint64(c) }

func count(toks []int32, vocab int) *counts {
	k := &counts{
		vocab: vocab, n: len(toks), c1: make([]int, vocab),
		c2: map[uint64]int{}, h1: map[uint64]int{},
		c3: map[uint64]int{}, h2: map[uint64]int{},
	}
	for i, w := range toks {
		k.c1[w]++
		if i >= 1 {
			k.c2[key2(toks[i-1], w)]++
			k.h1[uint64(toks[i-1])]++
		}
		if i >= 2 {
			k.c3[key3(toks[i-2], toks[i-1], w)]++
			k.h2[key2(toks[i-2], toks[i-1])]++
		}
	}
	return k
}

func (k *counts) p1(w int32) float64 {
	return float64(k.c1[w]+1) / float64(k.n+k.vocab)
}

func (k *counts) p2(u, w int32, l2 float64) float64 {
	p := k.p1(w)
	if h := k.h1[uint64(u)]; h > 0 {
		return l2*float64(k.c2[key2(u, w)])/float64(h) + (1-l2)*p
	}
	return p
}

func (k *counts) p3(u, v, w int32, l2, l3 float64) float64 {
	p := k.p2(v, w, l2)
	if h := k.h2[key2(u, v)]; h > 0 {
		return l3*float64(k.c3[key3(u, v, w)])/float64(h) + (1-l3)*p
	}
	return p
}

// score returns the bits and target bytes of order-n predictions for toks[from:]
// (from >= 2, so every order has its full context and all orders score the
// same targets).
func (k *counts) score(toks []int32, from int, order int, l2, l3 float64, bytes []int32) (bits float64, nbytes int64) {
	for i := from; i < len(toks); i++ {
		var p float64
		switch order {
		case 1:
			p = k.p1(toks[i])
		case 2:
			p = k.p2(toks[i-1], toks[i], l2)
		default:
			p = k.p3(toks[i-2], toks[i-1], toks[i], l2, l3)
		}
		bits -= math.Log2(p)
		nbytes += int64(bytes[toks[i]])
	}
	return
}

func bpb(bits float64, nbytes int64) float64 {
	if nbytes == 0 {
		return math.NaN()
	}
	return bits / float64(nbytes)
}

// Evaluate fits the three models on Train and scores Val. The interpolation
// weights are picked on the last 10% of Train with counts from the first 90%,
// so Val stays untouched until the final score.
func Evaluate(d *prep.Data) Result {
	v := d.Vocab()
	cut := len(d.Train) * 9 / 10
	l2, l3 := 0.5, 0.5
	if cut >= 2 && len(d.Train)-cut > 0 {
		fit := count(d.Train[:cut], v)
		// Held-out targets keep the context of the tokens just before them.
		from := max(cut, 2)
		best := math.Inf(1)
		for i := 1; i <= 9; i++ {
			l := float64(i) / 10
			b, _ := fit.score(d.Train, from, 2, l, 0, d.Bytes)
			if b < best {
				best, l2 = b, l
			}
		}
		best = math.Inf(1)
		for i := 1; i <= 9; i++ {
			l := float64(i) / 10
			b, _ := fit.score(d.Train, from, 3, l2, l, d.Bytes)
			if b < best {
				best, l3 = b, l
			}
		}
	}
	k := count(d.Train, v)
	var r Result
	b, n := k.score(d.Val, 2, 1, 0, 0, d.Bytes)
	r.Unigram = bpb(b, n)
	b, n = k.score(d.Val, 2, 2, l2, 0, d.Bytes)
	r.Bigram = bpb(b, n)
	b, n = k.score(d.Val, 2, 3, l2, l3, d.Bytes)
	r.Trigram = bpb(b, n)
	return r
}
