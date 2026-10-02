package sim

import (
	"math/rand"
	"testing"
)

func senseWorld(t *testing.T, seed int64, ticks int) *World {
	t.Helper()
	LoadConfig("config.yml")
	rand.Seed(seed)
	w := NewWorld()
	w.SerialBrains = true
	for range ticks {
		w.Update()
	}
	return w
}

// TestSenseCollectsThoughts checks that Sense asks for exactly the brains
// that think this tick, ants then monsters, writing into their own arrays.
func TestSenseCollectsThoughts(t *testing.T) {
	w := senseWorld(t, 3, 400)
	w.Sense()
	every := max(1, Cfg.ThinkEvery)
	want := map[*Ant]bool{}
	for _, a := range w.Ants {
		if a.Alive && (w.Tick+a.ID)%every == 0 {
			want[a] = true
		}
	}
	brained := 0
	for _, m := range w.Monsters {
		if m.Alive && m.Net != nil {
			brained++
		}
	}
	if len(w.Thoughts) != len(want)+brained {
		t.Fatalf("%d thoughts, want %d ants + %d monsters", len(w.Thoughts), len(want), brained)
	}
	for _, th := range w.Thoughts[:len(want)] {
		if len(th.In) != AntInputs || len(th.Out) != AntOutputs {
			t.Fatalf("ant thought with %d inputs and %d outputs", len(th.In), len(th.Out))
		}
	}
	for _, th := range w.Thoughts[len(want):] {
		if len(th.In) != MonInputs || len(th.Out) != MonOutputs {
			t.Fatalf("monster thought with %d inputs and %d outputs", len(th.In), len(th.Out))
		}
	}
	w.ThinkCPU()
	w.Act()
}

// TestActSkipsTheDead kills, between Sense and Act, every ant that was
// sensed and every monster's prey: Act must not step them nor chase them.
func TestActSkipsTheDead(t *testing.T) {
	w := senseWorld(t, 4, 300)
	w.Sense()
	w.ThinkCPU()
	ages := map[*Ant]int{}
	for _, a := range w.Ants {
		a.Alive = false
		ages[a] = a.Age
	}
	w.Act()
	for a, age := range ages {
		if a.Age != age {
			t.Fatalf("ant %d was stepped after dying", a.ID)
		}
	}
	for _, m := range w.Monsters {
		if m.prey != nil && !m.prey.Alive {
			t.Fatalf("monster %d still holds a dead prey after Act", m.ID)
		}
	}
}

// TestActPreyDropsDeadPrey kills a monster's prey between Sense and Act:
// the monster must not chase it, and one with no senses this tick gets no
// brain outputs.
func TestActPreyDropsDeadPrey(t *testing.T) {
	w := senseWorld(t, 4, 300)
	var m *Monster
	for range 3000 {
		w.Sense()
		for _, c := range w.Monsters {
			if c.Alive && c.prey != nil {
				m = c
				break
			}
		}
		if m != nil {
			break
		}
		w.ThinkCPU()
		w.Act()
	}
	if m == nil {
		t.Fatal("no monster found a prey")
	}
	m.prey.Alive = false
	prey, _, _ := w.actPrey(m)
	if prey != nil {
		t.Fatal("actPrey kept a dead prey")
	}
	if m.prey != nil {
		t.Fatal("actPrey left m.prey set")
	}
	m.sensedAt = w.Tick - 1
	if _, _, o := w.actPrey(m); o != nil {
		t.Fatal("unsensed monster got brain outputs")
	}
}
