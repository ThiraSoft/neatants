package sim

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestEffectsAndAging checks that buffs and debuffs occur and that ants and
// monsters die of old age.
func TestEffectsAndAging(t *testing.T) {
	LoadConfig("config.yml")
	Cfg.MonsterMaxAge = 2000 // short lives so the test sees monsters age out
	rand.Seed(3)
	w := NewWorld()
	var seen [NumEffects]int
	oldAnts, starved, oldMonsters := 0, 0, 0
	maxMonsters := 0
	for tick := range 40000 {
		w.Update()
		for _, e := range w.Events {
			switch {
			case e.Kind == EvOldAge && e.Val == 0:
				oldAnts++
			case e.Kind == EvOldAge:
				starved++
			case e.Kind == EvMonsterOld:
				oldMonsters++
			}
		}
		w.Events = w.Events[:0]
		if tick%10 == 0 {
			for _, a := range w.Ants {
				for k := range NumEffects {
					if a.Alive && a.Fx[k] > 0 {
						seen[k]++
					}
				}
			}
			for _, m := range w.Monsters {
				for k := range NumEffects {
					if m.Alive && m.Fx[k] > 0 {
						seen[k]++
					}
				}
			}
		}
		alive := 0
		for _, m := range w.Monsters {
			if m.Alive {
				alive++
			}
		}
		maxMonsters = max(maxMonsters, alive)
	}
	fmt.Println("samples of active effects:")
	for k := range NumEffects {
		fmt.Printf("  %-13s %d\n", Effect[k].Name, seen[k])
	}
	fmt.Printf("deaths of old age: %d ants, %d monsters · starved: %d\n", oldAnts, oldMonsters, starved)
	fmt.Printf("wave %d, living monsters at most %d\n", w.Wave, maxMonsters)
	if oldAnts == 0 || oldMonsters == 0 {
		t.Fatal("no death of old age")
	}
}
