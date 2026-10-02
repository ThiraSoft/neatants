// Package gpu runs the textevo evaluation on the card.
package gpu

import (
	_ "embed"
	"math"
	"slices"

	"github.com/ThiraSoft/neatants/neat"
)

// Capacities of the netrun kernel's shared memory. netrun is bound by how
// many workgroups are in flight, which the shared memory a workgroup takes
// decides, so it is built twice (see the go:generate lines in gen.go): a
// small variant just above what the largest champions of the first runs
// needed (799 values, 197 memory cells, 187 plastic links), which takes
// about a third of the shared memory of a big one, and the big one for the
// networks that outgrow it. The Max constants are those of the big one.
const (
	MaxValues  = 2048
	MaxNodes   = MaxValues
	MaxMemory  = 1024
	MaxPlastic = 1024
	// LevelEdges is the room for the products of one level, in both; Pack
	// splits a longer level into several, so only one node's edges must
	// fit.
	LevelEdges = 256

	smallValues  = 1024
	smallMemory  = 256
	smallPlastic = 256
)

//go:embed netrun.spv
var netrunSPV []byte

//go:embed netrun_big.spv
var netrunBigSPV []byte

// netClass is the netrun variant f runs in: 0 the small one, 1 the big one,
// -1 if it fits in neither.
func netClass(f *neat.Flat) int {
	_, back := backCopies(f)
	mem := 0
	for _, ni := range f.Order {
		if f.Kind[ni] == uint8(neat.Memory) {
			mem++
		}
		if f.Off[4*ni+4]-f.Off[4*ni] > LevelEdges {
			return -1
		}
	}
	values, pl := f.Nodes()+back, len(f.Plastic)
	switch {
	case values <= smallValues && mem <= smallMemory && pl <= smallPlastic:
		return 0
	case values <= MaxValues && mem <= MaxMemory && pl <= MaxPlastic:
		return 1
	}
	return -1
}

// Fits reports whether f fits in the shared memory of netrun.
func Fits(f *neat.Flat) bool { return netClass(f) >= 0 }

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

// recordPad zero words end a record, so that the kernel's clamped loads of a
// network with no edge or no node still read inside it.
const recordPad = 4

// Order words carry the node, its memory cell and whether it is a memory node.
const (
	orderMem       = 1 << 31
	orderCellShift = 16
)

// Pack appends the record of f to dst and returns the extended slice. The
// layout, in words from the start of the record:
//
//	n, nOrder, nLevels, nPlastic, inStart, inEnd, outStart, outEnd
//	kind[n]            kind | backCopy<<8
//	level[2(nLevels+1)] first slot and first edge of each level
//	order[nOrder]      node | cell<<16 | memory<<31, by level
//	off[4nOrder+1]     per slot and gate, the first edge; the last is the count
//	edges[2E]          from (bits 0-15 value index, 16-30 plastic index+1), weight bits
//	plastic[4P]        k, from, to, eta bits
//	pad[4]             zeros
//
// The edges are laid out in the order of the slots, not of the nodes, so that
// the edges of a level are contiguous: the kernel computes all their products
// at once, one lane an edge, and each node then adds its own in the order the
// CPU does. A level with more than LevelEdges edges is split in several,
// which is still a valid order since nodes of one level never read each
// other.
func Pack(dst []uint32, f *neat.Flat) []uint32 {
	n := f.Nodes()
	back, _ := backCopies(f)
	nOrder := len(f.Order)
	edgesOf := func(ni int32) int { return int(f.Off[4*ni+4] - f.Off[4*ni]) }
	levels := make([]int32, 1, len(f.LevelStart)+4)
	for l := 0; l+1 < len(f.LevelStart); l++ {
		e := 0
		for j := f.LevelStart[l]; j < f.LevelStart[l+1]; j++ {
			k := edgesOf(f.Order[j])
			if e+k > LevelEdges && j > levels[len(levels)-1] {
				levels = append(levels, j)
				e = 0
			}
			e += k
		}
		levels = append(levels, f.LevelStart[l+1])
	}
	nLevels := len(levels) - 1
	nEdges := 0
	for _, ni := range f.Order {
		nEdges += edgesOf(ni)
	}
	size := 8 + n + 2*(nLevels+1) + nOrder + 4*nOrder + 1 + 2*nEdges + 4*len(f.Plastic) + recordPad
	// Grow once, then fill by index: appending word by word costs more than
	// the rest of the record.
	at := len(dst)
	dst = slices.Grow(dst, size)[:at+size]
	w := dst[at:]
	w[0], w[1], w[2], w[3] = uint32(n), uint32(nOrder), uint32(nLevels), uint32(len(f.Plastic))
	w[4], w[5], w[6], w[7] = uint32(f.InStart), uint32(f.InEnd), uint32(f.OutStart), uint32(f.OutEnd)
	kindOff := 8
	levelOff := kindOff + n
	orderOff := levelOff + 2*(nLevels+1)
	offOff := orderOff + nOrder
	edgeOff := offOff + 4*nOrder + 1
	plOff := edgeOff + 2*nEdges
	for j, k := range f.Kind {
		w[kindOff+j] = uint32(k) | back[j]<<8
	}
	// moved[k] is where edge k of f lands.
	moved := make([]int32, len(f.From))
	for i := range moved {
		moved[i] = -1
	}
	cur, cell := 0, uint32(0)
	for j, ni := range f.Order {
		ow := uint32(ni)
		if f.Kind[ni] == uint8(neat.Memory) {
			ow |= orderMem | cell<<orderCellShift
			cell++
		}
		w[orderOff+j] = ow
		for g := range 4 {
			w[offOff+4*j+g] = uint32(cur)
			for k := f.Off[4*int(ni)+g]; k < f.Off[4*int(ni)+g+1]; k++ {
				x := f.From[k]
				if int(x) >= n { // a back edge reads the copy, which sits after the n values
					x = int32(n) + int32(back[int(x)-n]) - 1
				}
				w[edgeOff+2*cur] = uint32(x)
				w[edgeOff+2*cur+1] = math.Float32bits(f.Weight[k])
				moved[k] = int32(cur)
				cur++
			}
		}
	}
	w[offOff+4*nOrder] = uint32(cur)
	for i, s := range levels {
		w[levelOff+2*i] = uint32(s)
		w[levelOff+2*i+1] = w[offOff+4*int(s)]
	}
	for p, pl := range f.Plastic {
		k := moved[pl.K]
		w[edgeOff+2*int(k)] |= uint32(p+1) << 16
		i := plOff + 4*p
		w[i], w[i+1], w[i+2], w[i+3] = uint32(k), uint32(pl.From), uint32(pl.To), math.Float32bits(pl.Eta)
	}
	clear(w[size-recordPad:])
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
