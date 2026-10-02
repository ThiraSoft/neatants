// Package model holds the conventions shared by the CPU reference and the GPU
// kernels of textevo, so that both score a genome the same way.
package model

import (
	"math"
	"math/rand"

	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

const (
	// ValWindows and ValLen size the fixed validation set.
	ValWindows = 64
	ValLen     = 512
	// Warm is the number of ticks a window runs before it is scored, so the
	// recurrent state is not judged from cold.
	Warm = 16
)

// NewGenome returns a minimal genome mapping a dim embedding to a dim output,
// with one heritable trait: the logit scale. It has no state bank.
func NewGenome(id, dim int) *neat.Genome {
	return NewGenomeShape(id, Shape{dim, 0})
}

// Scale is the logit temperature s = 1 + 19*Traits[0], or its midpoint when
// the genome carries no trait.
func Scale(g *neat.Genome) float64 {
	if len(g.Traits) == 0 {
		return 10.5
	}
	return 1 + 19*g.Traits[0]
}

// LogitScale folds the 1/sqrt(dim) factor into the scale, in float32 as the
// kernels use it.
func LogitScale(g *neat.Genome, dim int) float32 {
	return float32(Scale(g) / math.Sqrt(float64(dim)))
}

// DrawStarts returns k window starts drawn uniformly so that a window of
// length ticks can read its last target, ids[start+length].
func DrawStarts(rng *rand.Rand, n, k, length int) []int {
	s := make([]int, k)
	for i := range s {
		s[i] = rng.Intn(n - length)
	}
	return s
}

// ValStarts returns ValWindows evenly spaced starts for windows of ValLen
// ticks, the last one still leaving room for its final target.
func ValStarts(n int) []int {
	s := make([]int, ValWindows)
	span := n - ValLen - 1
	for i := range s {
		s[i] = i * span / (ValWindows - 1)
	}
	return s
}

// WindowBytes is the byte length of the targets a set of windows scores,
// the denominator of bits per byte.
func WindowBytes(d *prep.Data, ids []int32, starts []int, length, warm int) int {
	n := 0
	for _, s := range starts {
		for t := warm; t < length; t++ {
			n += int(d.Bytes[ids[s+t+1]])
		}
	}
	return n
}
