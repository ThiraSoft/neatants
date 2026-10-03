package gpu

import (
	"slices"

	"github.com/ThiraSoft/neatants/neat"
)

// plasticPiece is how many plastic links one piece of the Oja step of back.comp
// covers: the two contributions of each must fit in LevelEdges.
const plasticPiece = 256

// PackLearn is Pack followed by the part of the record the backward pass
// needs, the one back.comp and the tape variant of netrun.comp read:
//
//	nMem, nBack, nPlasticPieces, 0
//	piece[2(P+1)]      per piece, its first group and its first perm entry
//	group[2(G+1)]      source, first perm entry; the last is a sentinel
//	perm[...]          entries of the groups, in order of the edges
//	pad[4]             zeros
//
// A piece is a level of Pack (levels first), then the pieces of the plastic
// links, plasticPiece of them each. The groups of a piece list the sources
// that receive a gradient from it, each with the entries that feed it: the
// edge of the level (counted from its first), or for a plastic piece
// 2*(link-first) for the link's source and +1 for its target. A source gets
// no gradient when it is the bias or an input, which are constants. The edges
// of a source are in the order of the record, which fixes the order the
// kernel adds them in.
//
// moved, of len(f.From), is filled as in pack.
func PackLearn(dst []uint32, f *neat.Flat, moved []int32) []uint32 {
	at := len(dst)
	dst = pack(dst, f, moved)
	rec := dst[at:]
	n, nOrder, nLevels, nPlastic := int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])
	outStart := rec[6]
	levelOff := 8 + n
	orderOff := levelOff + 2*(nLevels+1)
	offOff := orderOff + nOrder
	nEdges := int(rec[offOff+4*nOrder])
	edgeOff := offOff + 4*nOrder + 1
	_, nBack := backCopies(f)
	nMem := 0
	for _, ni := range f.Order {
		if f.Kind[ni] == uint8(neat.Memory) {
			nMem++
		}
	}
	nPP := (nPlastic + plasticPiece - 1) / plasticPiece

	type entry struct{ src, at uint32 }
	var pieces, groups, perm []uint32
	var es []entry
	// flush closes a piece: stable by source, one group per source.
	flush := func() {
		pieces = append(pieces, uint32(len(groups)/2), uint32(len(perm)))
		slices.SortStableFunc(es, func(a, b entry) int { return int(a.src) - int(b.src) })
		for i, e := range es {
			if i == 0 || es[i-1].src != e.src {
				groups = append(groups, e.src, uint32(len(perm)))
			}
			perm = append(perm, e.at)
		}
		es = es[:0]
	}
	for l := range nLevels {
		first := int(rec[levelOff+2*l+1])
		last := int(rec[levelOff+2*l+3])
		for k := first; k < last; k++ {
			if src := rec[edgeOff+2*k] & 0xFFFF; src >= outStart {
				es = append(es, entry{src, uint32(k - first)})
			}
		}
		flush()
	}
	plOff := edgeOff + 2*nEdges
	for pi := range nPP {
		q0 := pi * plasticPiece
		for q := q0; q < min(q0+plasticPiece, nPlastic); q++ {
			from, to := rec[plOff+4*q+1], rec[plOff+4*q+2]
			if from >= outStart {
				es = append(es, entry{from, uint32(2 * (q - q0))})
			}
			if to >= outStart {
				es = append(es, entry{to, uint32(2*(q-q0) + 1)})
			}
		}
		flush()
	}
	pieces = append(pieces, uint32(len(groups)/2), uint32(len(perm)))
	groups = append(groups, 0, uint32(len(perm)))
	dst = append(dst, uint32(nMem), uint32(nBack), uint32(nPP), 0)
	dst = append(dst, pieces...)
	dst = append(dst, groups...)
	dst = append(dst, perm...)
	return append(dst, 0, 0, 0, 0)
}

// learnShape reads from a record made by PackLearn what the tape and the
// gradient of its genome take: the edges, the plastic links, and the floats
// of one tick of the tape (see netrun.comp).
func learnShape(rec []uint32) (edges, plastic, stride int) {
	n, nOrder, nLevels, nPlastic := int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])
	levelOff := 8 + n
	offOff := levelOff + 2*(nLevels+1) + nOrder
	edges = int(rec[offOff+4*nOrder])
	ext := offOff + 4*nOrder + 1 + 2*edges + 4*nPlastic + recordPad
	nMem, nBack := int(rec[ext]), int(rec[ext+1])
	return edges, nPlastic, 2*n + nBack + 4*nMem + nPlastic
}
