package gpubrain

import (
	"math"

	"github.com/ThiraSoft/neatants/neat"
)

// Slot capacities. activate.comp is compiled with the same numbers
// (see the go:generate line in batch.go); TestLayoutMatchesShader keeps
// them in step.
const (
	MaxNodes   = 1024
	MaxEdges   = 4096
	MaxPlastic = 256

	// MaxValues is the room in shared memory for the values of a network:
	// its nodes, then the previous-tick values of the nodes some back edge
	// reads. It is kept small because it decides how many networks the card
	// runs at once; a network that needs more (about 600 is typical) runs on
	// the CPU.
	MaxValues = 1024
)

const (
	hdrWords  = 8
	kindOff   = hdrWords
	orderOff  = kindOff + MaxNodes
	levelOff  = orderOff + MaxNodes
	offOff    = levelOff + MaxNodes + 1
	edgeOff   = offOff + 4*MaxNodes + 1
	plOff     = edgeOff + 2*MaxEdges
	valsOff   = plOff + 4*MaxPlastic
	cellOff   = valsOff + MaxNodes
	SlotWords = cellOff + MaxNodes
)

// fits reports whether f fits in a slot.
func fits(f *neat.Flat) bool {
	_, back := backCopies(f)
	return f.Nodes() <= MaxNodes && len(f.From) <= MaxEdges && len(f.Plastic) <= MaxPlastic &&
		f.Nodes()+back <= MaxValues
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

// pack writes a newborn f into dst (SlotWords long): structure, starting
// weights, zero values and cells.
func pack(dst []uint32, f *neat.Flat) {
	clear(dst)
	dst[0] = uint32(f.Nodes())
	dst[1] = uint32(len(f.Order))
	dst[2] = uint32(len(f.LevelStart) - 1)
	dst[3] = uint32(len(f.Plastic))
	dst[4], dst[5], dst[6], dst[7] = uint32(f.InStart), uint32(f.InEnd), uint32(f.OutStart), uint32(f.OutEnd)
	// A kind word holds the kind, and above it where the node's previous
	// value is kept (see activate.comp).
	back, _ := backCopies(f)
	n := f.Nodes()
	for i, k := range f.Kind {
		dst[kindOff+i] = uint32(k) | back[i]<<8
	}
	for i, x := range f.Order {
		dst[orderOff+i] = uint32(x)
	}
	for i, x := range f.LevelStart {
		dst[levelOff+i] = uint32(x)
	}
	for i, x := range f.Off {
		dst[offOff+i] = uint32(x)
	}
	for i, x := range f.From {
		if int(x) >= n { // a back edge reads the copy, which sits after the n values
			x = int32(n) + int32(back[int(x)-n]) - 1
		}
		dst[edgeOff+2*i] = uint32(x)
		dst[edgeOff+2*i+1] = math.Float32bits(f.Weight[i])
	}
	for i, p := range f.Plastic {
		dst[plOff+4*i] = uint32(p.K)
		dst[plOff+4*i+1] = uint32(p.From)
		dst[plOff+4*i+2] = uint32(p.To)
		dst[plOff+4*i+3] = math.Float32bits(p.Eta)
	}
}
