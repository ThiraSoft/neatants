package sim

import (
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/neatants/neat"
)

// TestNewWorldsShareNothing builds two worlds from one read of the save and
// checks that no genome is shared and that changing one world leaves the
// other alone.
func TestNewWorldsShareNothing(t *testing.T) {
	LoadConfig("config.yml")
	old := SavePath
	SavePath = filepath.Join(t.TempDir(), "colonies.json")
	defer func() { SavePath = old }()
	ant := func(id int) *neat.Genome { return neat.NewGenomeWithHidden(id, AntInputs, AntOutputs, 4) }
	mon := func(id int) *neat.Genome { return neat.NewGenomeWithHidden(id, MonInputs, MonOutputs, 2) }
	var colonies []SavedColony
	for c := range 2 {
		colonies = append(colonies, SavedColony{
			Name: "c", Hall: []*neat.Genome{ant(10 * c), ant(10*c + 1)},
			Species: &Speciation{List: []*Species{{ID: 1, Rep: ant(10*c + 2)}}, NextID: 2},
		})
	}
	WriteSave(SaveData{Colonies: colonies, Monsters: []*neat.Genome{mon(50)},
		MonSpecies: &Speciation{List: []*Species{{ID: 1, Rep: mon(51)}}, NextID: 2}})
	ws := NewWorlds(2)
	seen := map[*neat.Genome]bool{}
	collect := func(w *World) (all []*neat.Genome) {
		for _, c := range w.Colonies {
			all = append(all, c.Hall...)
			for _, sp := range c.Species.List {
				all = append(all, sp.Rep)
			}
		}
		all = append(all, w.MonHall...)
		for _, sp := range w.MonSpecies.List {
			all = append(all, sp.Rep)
		}
		for _, a := range w.Ants {
			all = append(all, a.Genome)
		}
		return
	}
	for _, g := range collect(ws[0]) {
		seen[g] = true
	}
	n := 0
	for _, g := range collect(ws[1]) {
		if seen[g] {
			t.Fatal("a genome is shared between two worlds")
		}
		n++
	}
	if n == 0 {
		t.Fatal("the second world holds no genome")
	}
	var g0, g1 *neat.Genome
	for _, c := range ws[0].Colonies {
		if len(c.Hall) > 0 {
			g0 = c.Hall[0]
		}
	}
	for _, c := range ws[1].Colonies {
		if len(c.Hall) > 0 {
			g1 = c.Hall[0]
		}
	}
	if g0 == nil || g1 == nil || len(g0.Conns) == 0 {
		t.Skip("no hall to compare")
	}
	before := g1.Conns[0].Weight
	g0.Conns[0].Weight += 1
	if g1.Conns[0].Weight != before {
		t.Fatal("mutating a genome of one world changed the other")
	}
}
