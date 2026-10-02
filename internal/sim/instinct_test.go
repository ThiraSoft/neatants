package sim

import "testing"

// TestInstinctCap checks that no newborn exceeds instinct_max, whatever its lineage.
func TestInstinctCap(t *testing.T) {
	LoadConfig("config.yml")
	Cfg.InstinctMax = 0.4
	w := NewWorld()
	for _, c := range w.Colonies {
		for _, g := range c.Hall {
			g.Traits[TraitInstinct] = 0.95 // an evolved, very instinctive lineage
		}
	}
	for range 3000 {
		w.Update()
		w.Events = w.Events[:0]
	}
	for _, a := range w.Ants {
		if a.Alive && a.Age < 3000 && a.Instinct > Cfg.InstinctMax+1e-9 {
			t.Fatalf("ant born with an instinct of %.2f above the cap %.2f", a.Instinct, Cfg.InstinctMax)
		}
	}
}
