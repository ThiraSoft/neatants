package sim

import (
	"os"
	"testing"
)

// TestSaveSurvivesWipeout checks that a world where every colony has fallen
// still saves their lineages, and that an empty world never overwrites a save.
func TestSaveSurvivesWipeout(t *testing.T) {
	LoadConfig("config.yml")
	dir := t.TempDir()
	wd, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(wd)

	w := NewWorld()
	for range 3000 {
		w.Update()
	}
	for _, c := range w.Colonies {
		if len(c.Hall) == 0 {
			c.Hall = append(c.Hall, w.newGenome(c))
		}
		c.Alive = false
	}
	SaveWorld(w)
	pools := LoadSave().Pools
	if len(pools) != len(w.Colonies) {
		t.Fatalf("after total extinction, %d lineages saved instead of %d", len(pools), len(w.Colonies))
	}

	empty := &World{}
	SaveWorld(empty)
	if got := LoadSave().Pools; len(got) != len(pools) {
		t.Fatalf("an empty world overwrote the save (%d lineages)", len(got))
	}
}
