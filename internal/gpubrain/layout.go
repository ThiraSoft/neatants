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
)

const (
	hdrWords  = 8
	kindOff   = hdrWords
	orderOff  = kindOff + MaxNodes
	levelOff  = orderOff + MaxNodes
	offOff    = levelOff + MaxNodes + 1
	fromOff   = offOff + 4*MaxNodes + 1
	weightOff = fromOff + MaxEdges
	plOff     = weightOff + MaxEdges
	valsOff   = plOff + 4*MaxPlastic
	cellOff   = valsOff + MaxNodes
	SlotWords = cellOff + MaxNodes
)

// fits reports whether f fits in a slot.
func fits(f *neat.Flat) bool {
	return f.Nodes() <= MaxNodes && len(f.From) <= MaxEdges && len(f.Plastic) <= MaxPlastic
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
	for i, k := range f.Kind {
		dst[kindOff+i] = uint32(k)
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
		dst[fromOff+i] = uint32(x)
	}
	for i, x := range f.Weight {
		dst[weightOff+i] = math.Float32bits(x)
	}
	for i, p := range f.Plastic {
		dst[plOff+4*i] = uint32(p.K)
		dst[plOff+4*i+1] = uint32(p.From)
		dst[plOff+4*i+2] = uint32(p.To)
		dst[plOff+4*i+3] = math.Float32bits(p.Eta)
	}
}
