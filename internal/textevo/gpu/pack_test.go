package gpu

import (
	"math"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// unpack reads a record back: for each slot its node and, per gate, the
// value index, plastic tag and weight bits of its edges, in record order.
type unpacked struct {
	nLevels int
	levels  [][2]int // first slot and first edge of each level, and the end
	order   []uint32
	gates   [][4][][2]uint32
	plastic [][4]uint32
	pad     []uint32
}

func unpack(t *testing.T, w []uint32) unpacked {
	t.Helper()
	n, nOrder, nLevels, nPlastic := int(w[0]), int(w[1]), int(w[2]), int(w[3])
	levelOff := 8 + n
	orderOff := levelOff + 2*(nLevels+1)
	offOff := orderOff + nOrder
	nE := int(w[offOff+4*nOrder])
	edgeOff := offOff + 4*nOrder + 1
	plOff := edgeOff + 2*nE
	if len(w) != plOff+4*nPlastic+recordPad {
		t.Fatalf("record is %d words, want %d", len(w), plOff+4*nPlastic+recordPad)
	}
	u := unpacked{nLevels: nLevels, pad: w[plOff+4*nPlastic:]}
	for l := range nLevels + 1 {
		u.levels = append(u.levels, [2]int{int(w[levelOff+2*l]), int(w[levelOff+2*l+1])})
	}
	for j := range nOrder {
		u.order = append(u.order, w[orderOff+j])
		var gs [4][][2]uint32
		for g := range 4 {
			for k := w[offOff+4*j+g]; k < w[offOff+4*j+g+1]; k++ {
				gs[g] = append(gs[g], [2]uint32{w[edgeOff+2*int(k)], w[edgeOff+2*int(k)+1]})
			}
		}
		u.gates = append(u.gates, gs)
	}
	for p := range nPlastic {
		u.plastic = append(u.plastic, [4]uint32(w[plOff+4*p:plOff+4*p+4]))
	}
	return u
}

// checkRecord compares a record with the network it was packed from.
func checkRecord(t *testing.T, f *neat.Flat, w []uint32) unpacked {
	t.Helper()
	n := f.Nodes()
	if int(w[0]) != n || int(w[1]) != len(f.Order) || int(w[3]) != len(f.Plastic) {
		t.Fatalf("bad header %v", w[:4])
	}
	if int32(w[4]) != f.InStart || int32(w[5]) != f.InEnd || int32(w[6]) != f.OutStart || int32(w[7]) != f.OutEnd {
		t.Fatalf("bad io ranges %v", w[4:8])
	}
	u := unpack(t, w)
	for _, x := range u.pad {
		if x != 0 {
			t.Fatal("padding is not zero")
		}
	}
	// The levels cover the slots in order, each inside one level of f, and
	// hold at most LevelEdges edges unless they are a single node.
	level := make([]int, len(f.Order))
	for l := 0; l+1 < len(f.LevelStart); l++ {
		for j := f.LevelStart[l]; j < f.LevelStart[l+1]; j++ {
			level[j] = l
		}
	}
	if u.levels[0][0] != 0 || u.levels[u.nLevels][0] != len(f.Order) {
		t.Fatalf("levels do not cover the order: %v", u.levels)
	}
	for l := range u.nLevels {
		s, e := u.levels[l][0], u.levels[l+1][0]
		if s >= e || level[s] != level[e-1] {
			t.Fatalf("level %d [%d,%d) is empty or crosses a level of the network", l, s, e)
		}
		if edges := u.levels[l+1][1] - u.levels[l][1]; edges > LevelEdges && e-s > 1 {
			t.Fatalf("level %d holds %d edges", l, edges)
		}
	}
	back, _ := backCopies(f)
	cells := 0
	tag := map[int]int{} // edge index of f -> plastic index + 1
	for p, pl := range f.Plastic {
		tag[int(pl.K)] = p + 1
	}
	for j, ni := range f.Order {
		ow := u.order[j]
		mem := f.Kind[ni] == uint8(neat.Memory)
		if int32(ow&0xFFFF) != ni || (ow&orderMem != 0) != mem {
			t.Fatalf("slot %d: order word %#x for node %d", j, ow, ni)
		}
		if mem {
			if int(ow>>orderCellShift&0x7FFF) != cells {
				t.Fatalf("slot %d: cell %d, want %d", j, ow>>orderCellShift&0x7FFF, cells)
			}
			cells++
		}
		for g := range 4 {
			lo, hi := f.Off[4*int(ni)+g], f.Off[4*int(ni)+g+1]
			if len(u.gates[j][g]) != int(hi-lo) {
				t.Fatalf("slot %d gate %d: %d edges, want %d", j, g, len(u.gates[j][g]), hi-lo)
			}
			for i, e := range u.gates[j][g] {
				k := int(lo) + i
				want := f.From[k]
				if int(want) >= n {
					want = int32(n) + int32(back[int(want)-n]) - 1
				}
				if e[0]&0xFFFF != uint32(want) || int(e[0]>>16) != tag[k] || e[1] != math.Float32bits(f.Weight[k]) {
					t.Fatalf("slot %d gate %d edge %d: %#x %#x", j, g, i, e[0], e[1])
				}
			}
		}
	}
	for p, pl := range f.Plastic {
		r := u.plastic[p]
		if int32(r[1]) != pl.From || int32(r[2]) != pl.To || r[3] != math.Float32bits(pl.Eta) {
			t.Fatalf("plastic %d record differs", p)
		}
		edgeOff := 8 + n + 2*(u.nLevels+1) + 5*len(f.Order) + 1
		if int(w[edgeOff+2*int(r[0])]>>16) != p+1 || w[edgeOff+2*int(r[0])+1] != math.Float32bits(f.Weight[pl.K]) {
			t.Fatalf("plastic %d points at edge %d, which is not its own", p, r[0])
		}
	}
	return u
}

func TestPackRecord(t *testing.T) {
	f := model.Grown(1, 16, 500).BuildNetwork().Flat()
	if len(f.Plastic) == 0 {
		t.Fatal("test genome has no plastic link")
	}
	w := Pack(nil, f)
	checkRecord(t, f, w)
	// A second record is appended, not overwritten.
	if w2 := Pack(w, f); len(w2) != 2*len(w) {
		t.Fatalf("append gave %d words", len(w2))
	}
}

// A level with more edges than LevelEdges is split, and the record stays
// the same network.
func TestPackSplitsLongLevels(t *testing.T) {
	f := model.Grown(1, 300, 200).BuildNetwork().Flat()
	u := checkRecord(t, f, Pack(nil, f))
	if u.nLevels <= len(f.LevelStart)-1 {
		t.Fatalf("%d levels for %d in the network: nothing was split", u.nLevels, len(f.LevelStart)-1)
	}
}

func TestFits(t *testing.T) {
	if Fits(&neat.Flat{Kind: make([]uint8, MaxNodes+1)}) {
		t.Fatal("too many nodes must not fit")
	}
	if !Fits(&neat.Flat{Kind: make([]uint8, 10)}) {
		t.Fatal("a small network must fit")
	}
	// One node reading more edges than a level holds cannot be split.
	wide := &neat.Flat{Kind: make([]uint8, 3), Order: []int32{2}, Off: make([]int32, 13), From: make([]int32, LevelEdges+1)}
	for s := 9; s < 13; s++ {
		wide.Off[s] = LevelEdges + 1
	}
	if Fits(wide) {
		t.Fatal("a node with more edges than LevelEdges must not fit")
	}
	wide.From, wide.Off[9], wide.Off[10], wide.Off[11], wide.Off[12] = wide.From[:LevelEdges], LevelEdges, LevelEdges, LevelEdges, LevelEdges
	if !Fits(wide) {
		t.Fatal("a node with LevelEdges edges must fit")
	}
}
