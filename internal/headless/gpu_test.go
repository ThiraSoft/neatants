package headless

import (
	"math/rand"
	"os"
	"syscall"
	"testing"
	"time"

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

// TestPipelineKeepsWorldsInStep runs the A/B pipeline and checks that every
// world advanced by exactly the number of steps, and that drain, called with
// a dispatch in flight, finishes it without deadlock.
func TestPipelineKeepsWorldsInStep(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	rand.Seed(2)
	worlds := make([]*sim.World, 5)
	for i := range worlds {
		worlds[i] = sim.NewWorld()
	}
	p, err := newPipeline(d, worlds)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	if err := p.prime(); err != nil {
		t.Fatal(err)
	}
	for range 300 {
		if err := p.step(); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.drain(); err != nil {
		t.Fatal(err)
	}
	// prime runs the first Sense and drain the last Act: 301 ticks.
	for i, w := range worlds {
		if w.Tick != 301 {
			t.Fatalf("world %d at tick %d after prime, 300 steps and drain", i, w.Tick)
		}
	}
}

// TestRunWorldsHonoursTicks checks that the pipeline's prime and drain do not
// add ticks: 7 ticks in epochs of 3 must leave every world at tick 7.
func TestRunWorldsHonoursTicks(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	rand.Seed(3)
	worlds := make([]*sim.World, 4)
	for i := range worlds {
		worlds[i] = sim.NewWorld()
	}
	runWorlds(worlds, 7, 3, time.Hour, true)
	for i, w := range worlds {
		if w.Tick != 7 {
			t.Fatalf("world %d at tick %d, want 7", i, w.Tick)
		}
	}
}

// TestRunWorldsStopsAndSaves runs the GPU runner for a fixed number of ticks
// with an epoch shorter than the run: it must return, with the save written.
// WriteSave skips colonies without a hall of fame, so the run must be long
// enough for some ants to die and fill one.
func TestRunWorldsStopsAndSaves(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	waitReturn(t, func() { RunWorlds(4, 6000, 500, time.Hour, true) })
	if _, err := os.Stat(sim.SavePath); err != nil {
		t.Fatalf("no save after the run: %v", err)
	}
}

// TestRunWorldsInterrupted sends SIGINT to an unbounded run, as Ctrl+C does:
// the pipeline must drain, save and return, for several worlds and for one.
func TestRunWorldsInterrupted(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	d.Close()
	sim.LoadConfig("config.yml")
	for _, n := range []int{4, 1} {
		sim.SavePath = t.TempDir() + "/colonies.json"
		go func() {
			time.Sleep(12 * time.Second) // long enough for a hall of fame to fill
			syscall.Kill(os.Getpid(), syscall.SIGINT)
		}()
		waitReturn(t, func() { RunWorlds(n, 0, 100000, time.Hour, true) })
		if _, err := os.Stat(sim.SavePath); err != nil {
			t.Fatalf("%d worlds: no save after Ctrl+C: %v", n, err)
		}
	}
}

// waitReturn fails the test if f does not return within five minutes.
func waitReturn(t *testing.T, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		f()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Minute):
		t.Fatal("RunWorlds did not return")
	}
}
