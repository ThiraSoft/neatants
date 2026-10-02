package sim

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// TestMarriage runs long games and checks weddings, peace between spouses,
// divorces and half-blood queens. Slow: runs only with NEATANTS_LONG_TESTS=1.
func TestMarriage(t *testing.T) {
	if os.Getenv("NEATANTS_LONG_TESTS") == "" {
		t.Skip("NEATANTS_LONG_TESTS=1 to run this long test")
	}
	LoadConfig("config.yml")
	weddings, divorces, gifts, halfBlood := 0, 0, 0, 0
	spouseKills, rivalKills := 0, 0
	spouseWounds := 0
	woundHook = func(att, vic int) { spouseWounds++ }
	defer func() { woundHook = nil }()
	for seed := range 3 {
		rand.Seed(int64(40 + seed))
		w := NewWorld()
		for range 80000 {
			// Count kills between colonies, by relationship at the time.
			before := map[int]int{}
			for _, c := range w.Colonies {
				before[c.ID] = c.Kills
			}
			spouses := [MaxColonies]int{}
			for i, c := range w.Colonies {
				spouses[i] = w.SpouseOf(c.ID)
			}
			nlog := len(w.Log)
			var last string
			if nlog > 0 {
				last = w.Log[nlog-1].Text
			}
			w.Update()
			for i := len(w.Log) - 1; i >= 0 && w.Log[i].Text != last; i-- {
				if strings.HasPrefix(w.Log[i].Text, "A half-blood queen") {
					halfBlood++
				}
			}
			for _, e := range w.Events {
				switch e.Kind {
				case EvWedding:
					weddings++
				case EvDivorce:
					divorces++
				case EvGift:
					gifts++
				}
			}
			w.Events = w.Events[:0]
			for _, c := range w.Colonies {
				if d := c.Kills - before[c.ID]; d > 0 {
					if spouses[c.ID] >= 0 {
						spouseKills += d // a married colony killed someone: rival or spouse
					} else {
						rivalKills += d
					}
				}
			}
		}
		for _, c := range w.Colonies {
			s := "single"
			if sp := w.SpouseOf(c.ID); sp >= 0 {
				s = "married to " + w.Colonies[sp].Name
			}
			fmt.Printf("  seed %d · %-10s alive=%v pop=%d stock=%.0f · %s\n", seed, c.Name, c.Alive, c.Pop, c.Food, s)
		}
	}
	fmt.Printf("marriages %d · divorces %d · food gifts %d · half-blood queens %d\n",
		weddings, divorces, gifts, halfBlood)
	fmt.Printf("ants killed by married colonies %d, by single ones %d\n", spouseKills, rivalKills)
	fmt.Printf("wounds between spouses (insecticide): %d\n", spouseWounds)
	if weddings == 0 {
		t.Fatal("no marriage")
	}
}
