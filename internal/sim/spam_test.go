package sim

import (
	"math/rand"
	"testing"
)

// TestSpellSpamCosts checks that a spell cast into the void costs the caster
// (cooldown, energy, fitness) while one that strikes a foe does not.
func TestSpellSpamCosts(t *testing.T) {
	LoadConfig("config.yml")
	Cfg.BrainMode = "full"
	rand.Seed(7)
	w := NewWorld()
	w.Monsters = w.Monsters[:0]
	var a, foe *Ant
	for _, o := range w.Ants {
		switch {
		case a == nil && o.Colony == 0:
			a = o
		case foe == nil && o.Colony == 1:
			foe = o
		default:
			o.Alive = false
		}
	}
	foe.Alive = false
	a.Pos = Vec2{WorldW / 2, WorldH / 2}
	a.Element, a.Mana, a.Energy = ElemFrost, 1, 1
	w.rebuildGrids()

	w.castSpell(a)
	if a.Casts != 1 || a.Wasted != 1 {
		t.Fatalf("spell into the void: %d cast, %d wasted, expected 1 and 1", a.Casts, a.Wasted)
	}
	if a.CastCD != wasteCD {
		t.Fatalf("cooldown after a wasted spell: %d, expected %d", a.CastCD, wasteCD)
	}
	if a.Energy > 1-wasteEnergy+1e-9 {
		t.Fatalf("a wasted spell should cost energy (%.3f left)", a.Energy)
	}

	foe.Alive = true
	foe.Pos = a.Pos.Add(Vec2{20, 0})
	a.Mana, a.CastCD = 1, 0
	w.rebuildGrids()
	w.castSpell(a)
	if a.Casts != 2 || a.Wasted != 1 {
		t.Fatalf("spell on an enemy: %d cast, %d wasted, expected 2 and 1", a.Casts, a.Wasted)
	}
	if a.CastCD != castCD {
		t.Fatalf("cooldown after a successful spell: %d, expected %d", a.CastCD, castCD)
	}

	clean := *a
	clean.Wasted = 0
	if individualFitness(a, 1) >= individualFitness(&clean, 1) {
		t.Fatal("wasted spells should lower fitness")
	}
}
