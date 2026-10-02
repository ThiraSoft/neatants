package headless

import (
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/sim"
)

func TestMain(m *testing.M) {
	if err := os.Chdir("../.."); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

// TestGroupTicksLikeCPU steps worlds through the GPU group for a few hundred
// ticks: they must keep living (ants born and dying, monsters thinking)
// without errors, and the batch must have uploaded and freed networks.
func TestGroupTicksLikeCPU(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	rand.Seed(1)
	worlds := []*sim.World{sim.NewWorld(), sim.NewWorld(), sim.NewWorld()}
	g, err := newGroup(d, worlds)
	if err != nil {
		t.Fatal(err)
	}
	defer g.close()
	for range 600 {
		g.sense()
		if err := g.start(); err != nil {
			t.Fatal(err)
		}
		if err := g.finish(); err != nil {
			t.Fatal(err)
		}
		g.act()
	}
	up, freed, _ := g.batch.Stats()
	if up == 0 || freed == 0 {
		t.Fatalf("uploaded %d, freed %d networks in 600 ticks", up, freed)
	}
	for i, w := range worlds {
		if w.Tick != 600 || len(w.Ants) == 0 {
			t.Fatalf("world %d at tick %d with %d ants", i, w.Tick, len(w.Ants))
		}
	}
}
