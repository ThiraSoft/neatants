// Package gpu runs the textevo evaluation on the card.
package gpu

import (
	_ "embed"
	"math"

	"github.com/ThiraSoft/neatants/neat"
)

// Capacities of the netrun kernel's shared memory; netrun.spv is compiled
// with the same numbers (see the go:generate line in gen.go).
const (
	MaxNodes   = 2048
	MaxValues  = 2048
	MaxPlastic = 1024
)

//go:embed netrun.spv
var netrunSPV []byte

// Fits reports whether f fits in the shared memory of netrun.
func Fits(f *neat.Flat) bool {
	_, back := backCopies(f)
	return f.Nodes() <= MaxNodes && f.Nodes()+back <= MaxValues && len(f.Plastic) <= MaxPlastic
}

// backCopies numbers the nodes whose previous-tick value some edge reads.
// Flat addresses those values at n+node, 2n floats for n nodes of which only
// a few are ever read; the kernel keeps just those, right after the n current
// values. at[node] is the 1-based place of node's copy (0: none) and count is
// how many there are.
func backCopies(f *neat.Flat) (at []uint32, count int) {
	n := f.Nodes()
	at = make([]uint32, n)
	for _, x := range f.From {
		if int(x) >= n {
			at[int(x)-n] = 1
		}
	}
	for i, a := range at {
		if a != 0 {
			count++
			at[i] = uint32(count)
		}
	}
	return at, count
}

// Pack appends the record of f to dst and returns the extended slice. The
// layout, in words from the start of the record:
//
//	n, nOrder, nLevels, nPlastic, inStart, inEnd, outStart, outEnd
//	kind[n]            kind | backCopy<<8
//	levelStart[nLevels+1]
//	order[nOrder]
//	off[4n+1]          off[4n] is the edge count
//	edges[2E]          from (bits 0-15 value index, 16-30 plastic index+1), weight bits
//	plastic[4P]        k, from, to, eta bits
func Pack(dst []uint32, f *neat.Flat) []uint32 {
	n := f.Nodes()
	back, _ := backCopies(f)
	dst = append(dst, uint32(n), uint32(len(f.Order)), uint32(len(f.LevelStart)-1), uint32(len(f.Plastic)),
		uint32(f.InStart), uint32(f.InEnd), uint32(f.OutStart), uint32(f.OutEnd))
	for i, k := range f.Kind {
		dst = append(dst, uint32(k)|back[i]<<8)
	}
	for _, x := range f.LevelStart {
		dst = append(dst, uint32(x))
	}
	for _, x := range f.Order {
		dst = append(dst, uint32(x))
	}
	for _, x := range f.Off {
		dst = append(dst, uint32(x))
	}
	edges := len(dst)
	for i, x := range f.From {
		if int(x) >= n { // a back edge reads the copy, which sits after the n values
			x = int32(n) + int32(back[int(x)-n]) - 1
		}
		dst = append(dst, uint32(x), math.Float32bits(f.Weight[i]))
	}
	for p, pl := range f.Plastic {
		dst[edges+2*int(pl.K)] |= uint32(p+1) << 16
	}
	for _, p := range f.Plastic {
		dst = append(dst, uint32(p.K), uint32(p.From), uint32(p.To), math.Float32bits(p.Eta))
	}
	return dst
}

// halfToFloat decodes an IEEE binary16 value.
func halfToFloat(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	e := int(h>>10) & 0x1F
	m := float32(h & 0x3FF)
	switch e {
	case 0:
		return sign * m * (1.0 / (1 << 24))
	case 31:
		if m == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	return sign * (1 + m/1024) * float32(math.Ldexp(1, e-15))
}

// floatToHalf encodes x as an IEEE binary16 value, rounding to nearest even.
// Callers keep |x| well under the half range, so larger values are clamped.
func floatToHalf(x float32) uint16 {
	b := math.Float32bits(x)
	sign := uint16(b>>16) & 0x8000
	e := int(b>>23&0xFF) - 127 + 15
	m := b & 0x7FFFFF
	switch {
	case e >= 31:
		return sign | 0x7BFF
	case e <= 0:
		if e < -10 {
			return sign
		}
		// Subnormal half: shift the implicit one in, then round to even.
		m |= 0x800000
		shift := uint(14 - e)
		h := m >> shift
		rem := m & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(e)<<10 | m>>13
	rem := m & 0x1FFF
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++ // a carry into the exponent is the correct rounding up
	}
	if h >= 0x7C00 {
		h = 0x7BFF
	}
	return sign | uint16(h)
}
