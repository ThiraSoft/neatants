package model

import (
	"math/rand"
	"testing"
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
