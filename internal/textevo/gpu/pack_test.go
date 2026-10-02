package gpu

import (
	"math"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

func TestPackRecord(t *testing.T) {
	f := model.Grown(1, 16, 500).BuildNetwork().Flat()
	w := Pack(nil, f)
	n := f.Nodes()
	if int(w[0]) != n || int(w[1]) != len(f.Order) || int(w[2]) != len(f.LevelStart)-1 || int(w[3]) != len(f.Plastic) {
		t.Fatalf("bad header %v", w[:4])
	}
	if int32(w[4]) != f.InStart || int32(w[5]) != f.InEnd || int32(w[6]) != f.OutStart || int32(w[7]) != f.OutEnd {
		t.Fatalf("bad io ranges %v", w[4:8])
	}
	levelOff := 8 + n
	orderOff := levelOff + len(f.LevelStart)
	offOff := orderOff + len(f.Order)
	nE := int(w[offOff+4*n])
	if nE != len(f.From) {
		t.Fatalf("edge count %d, want %d", nE, len(f.From))
	}
	edgeOff := offOff + 4*n + 1
	plOff := edgeOff + 2*nE
	if len(w) != plOff+4*len(f.Plastic) {
		t.Fatalf("record is %d words, want %d", len(w), plOff+4*len(f.Plastic))
	}
	for k := range nE {
		from := int(w[edgeOff+2*k] & 0xFFFF)
		if int(f.From[k]) < n && from != int(f.From[k]) {
			t.Fatalf("edge %d reads %d, want %d", k, from, f.From[k])
		}
		if int(f.From[k]) >= n && from < n {
			t.Fatalf("back edge %d reads %d, below n", k, from)
		}
		if w[edgeOff+2*k+1] != math.Float32bits(f.Weight[k]) {
			t.Fatalf("edge %d weight bits differ", k)
		}
	}
	if len(f.Plastic) == 0 {
		t.Fatal("test genome has no plastic link")
	}
	for p, pl := range f.Plastic {
		if got := int(w[edgeOff+2*int(pl.K)]>>16) & 0x7FFF; got != p+1 {
			t.Fatalf("plastic %d: edge tag %d", p, got)
		}
		if int32(w[plOff+4*p]) != pl.K || int32(w[plOff+4*p+1]) != pl.From || int32(w[plOff+4*p+2]) != pl.To {
			t.Fatalf("plastic %d record differs", p)
		}
	}
	// A second record is appended, not overwritten.
	if w2 := Pack(w, f); len(w2) != 2*len(w) {
		t.Fatalf("append gave %d words", len(w2))
	}
}

func TestFits(t *testing.T) {
	if Fits(&neat.Flat{Kind: make([]uint8, MaxNodes+1)}) {
		t.Fatal("too many nodes must not fit")
	}
	if !Fits(&neat.Flat{Kind: make([]uint8, 10)}) {
		t.Fatal("a small network must fit")
	}
}
