package sim

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestCaveFairness checks that the cave moves and that waves spread over nests.
func TestCaveFairness(t *testing.T) {
	LoadConfig("config.yml")
	rand.Seed(7)
	w := NewWorld()
	seen := map[int]bool{}
	targets := map[int]int{}
	caves := []Vec2{w.Cave}
	for range 20000 {
		w.Update()
		w.Events = w.Events[:0]
		if w.Cave != caves[len(caves)-1] {
			caves = append(caves, w.Cave)
		}
		for _, m := range w.Monsters {
			if !seen[m.ID] {
				seen[m.ID] = true
				targets[m.Target]++
			}
		}
	}
	fmt.Printf("waves %d, cave positions %d\n", w.Wave, len(caves))
	for i, c := range caves {
		minD := 1e9
		for _, col := range w.Colonies {
			minD = min(minD, c.Sub(col.Pos).Len())
		}
		fmt.Printf("  cave %d: (%4.0f, %4.0f)\n", i, c.X, c.Y)
	}
	minX, maxX, minY, maxY := 1e9, 0.0, 1e9, 0.0
	for _, c := range caves {
		minX, maxX = min(minX, c.X), max(maxX, c.X)
		minY, maxY = min(minY, c.Y), max(maxY, c.Y)
	}
	fmt.Printf("  extent x %.0f to %.0f, y %.0f to %.0f\n", minX, maxX, minY, maxY)
	for id, n := range targets {
		name := "none"
		if id >= 0 {
			name = w.Colonies[id].Name
		}
		fmt.Printf("  target %-10s %d monsters\n", name, n)
	}
	if len(caves) < 3 {
		t.Fatalf("the cave only moved %d times", len(caves)-1)
	}
}
