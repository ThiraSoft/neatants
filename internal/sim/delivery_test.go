package sim

import (
	"os"
	"testing"
)

// Food that lands near a living nest is lost: neither on the ground nor in
// the reserves, so the colony has to walk to eat.
func TestNoFoodNearNests(t *testing.T) {
	LoadConfig("config.yml")
	wd, _ := os.Getwd()
	os.Chdir(t.TempDir())
	defer os.Chdir(wd)
	w := NewWorld()
	c := w.Colonies[0]
	before, items := c.Food, len(w.Foods)
	w.scatterFood(c.Pos, 10, 2)
	for _, f := range w.Foods[items:] {
		if f.Pos.Sub(c.Pos).Len() < FoodNestGap {
			t.Fatalf("food dropped %.0f from the nest", f.Pos.Sub(c.Pos).Len())
		}
	}
	if got := c.Food - before; got != 0 {
		t.Fatalf("stock %+.1f, expected no change", got)
	}
}

// A delivery only counts when the food was picked up away from the nest,
// not when the ant turns on the spot next to a carcass.
func TestNoDeliveryForFoodPickedUpAtTheNest(t *testing.T) {
	LoadConfig("config.yml")
	Cfg.BrainMode = "hybrid" // fresh brains alone rarely bring food from FoodNestGap away
	wd, _ := os.Getwd()
	os.Chdir(t.TempDir())
	defer os.Chdir(wd)
	w := NewWorld()
	type track struct {
		carry bool
		del   int
		from  Vec2
	}
	s := map[*Ant]*track{}
	total := 0
	for range 20000 {
		prev := map[*Ant]Vec2{}
		for _, a := range w.Ants {
			prev[a] = a.Pos
		}
		w.Update()
		w.Events = w.Events[:0]
		for _, a := range w.Ants {
			x, ok := s[a]
			if !ok {
				s[a] = &track{carry: a.Carrying, del: a.Delivered, from: a.Pos}
				continue
			}
			if a.Delivered > x.del {
				total++
				from := x.from
				if !x.carry {
					from = prev[a]
				}
				if d := from.Sub(w.Colonies[a.Colony].Pos).Len(); d < DeliveryMinDist-12 {
					t.Fatalf("delivery counted for food picked up %.0f from the nest", d)
				}
				x.del = a.Delivered
			}
			if a.Carrying && !x.carry {
				x.from = a.Pos
			}
			x.carry = a.Carrying
		}
	}
	if total == 0 {
		t.Fatal("no delivery in 20000 ticks")
	}
}
