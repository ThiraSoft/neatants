package model

import (
	"math/rand"
	"testing"

	"github.com/ThiraSoft/neatants/neat"
)

func TestDrawStartsLeaveRoomForTheTarget(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for range 1000 {
		for _, s := range DrawStarts(rng, 200, 4, 128) {
			if s < 0 || s+128 >= 200 { // reads ids[s .. s+128] inclusive
				t.Fatalf("start %d", s)
			}
		}
	}
	if s := DrawStarts(rng, 129, 2, 128); s[0] != 0 || s[1] != 0 {
		t.Fatalf("tight corpus: %v", s)
	}
}

func TestValStarts(t *testing.T) {
	s := ValStarts(100000)
	if len(s) != ValWindows || s[0] != 0 || s[len(s)-1]+ValLen >= 100000 {
		t.Fatalf("%v", s)
	}
}

func TestScale(t *testing.T) {
	g := NewGenome(1, 8)
	g.Traits[0] = 0
	if Scale(g) != 1 {
		t.Fatal(Scale(g))
	}
	g.Traits[0] = 1
	if Scale(g) != 20 {
		t.Fatal(Scale(g))
	}
}

// A dense start (neat.MinimalLinks = dim) must give every output all the
// inputs and the bias, so the -links flag of neattext reaches the genome.
func TestNewGenomeHonoursMinimalLinks(t *testing.T) {
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	for _, k := range []int{1, 5, 16} {
		neat.MinimalLinks = k
		g := NewGenome(1, 16)
		reads := map[int]int{}
		for _, c := range g.Conns {
			if c.In != 0 {
				reads[c.Out]++
			}
		}
		if len(reads) != 16 || len(g.Conns) != 16*(k+1) {
			t.Fatalf("links %d: %d outputs read inputs, %d connections", k, len(reads), len(g.Conns))
		}
		for out, n := range reads {
			if n != k {
				t.Fatalf("links %d: output %d reads %d inputs", k, out, n)
			}
		}
	}
}
