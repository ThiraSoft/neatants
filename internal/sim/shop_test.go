package sim

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestShopping checks that colonies buy insecticide and spray monsters.
func TestShopping(t *testing.T) {
	LoadConfig("config.yml")
	Cfg.BrainMode = "hybrid" // walking to the shop is a scripted errand, off in full mode
	rand.Seed(5)
	w := NewWorld()
	buys, sprays := 0, 0
	for range 40000 {
		w.Update()
		for _, e := range w.Events {
			switch e.Kind {
			case EvShopBuy:
				buys++
			case EvSpray:
				sprays++
			}
		}
		w.Events = w.Events[:0]
	}
	fmt.Printf("Wallmart at (%.0f, %.0f) · %d purchases · %d sprays\n", w.Shop.X, w.Shop.Y, buys, sprays)
	for _, c := range w.Colonies {
		fmt.Printf("  %-10s stock %.0f · %d bombs · %d ants\n", c.Name, c.Food, c.Cans, c.Pop)
	}
	if buys == 0 || sprays == 0 {
		t.Fatal("nobody shopped or sprayed")
	}
}
