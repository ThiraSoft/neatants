package headless

import (
	"math/rand"
	"os"
	"sync/atomic"
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

// TestSchedulerTicksLikeCPU runs worlds through one GPU group for a few
// hundred ticks: they must keep living (ants born and dying, monsters
// thinking) without errors, and the batch must have uploaded and freed
// networks.
func TestSchedulerTicksLikeCPU(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	rand.Seed(1)
	worlds := []*sim.World{sim.NewWorld(), sim.NewWorld(), sim.NewWorld()}
	var stopped atomic.Bool
	s, err := newScheduler(d, worlds, 1, &stopped)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if _, err := s.run(600); err != nil {
		t.Fatal(err)
	}
	up, freed, _ := s.groups[0].batch.Stats()
	if up == 0 || freed == 0 {
		t.Fatalf("uploaded %d, freed %d networks in 600 ticks", up, freed)
	}
	for i, w := range worlds {
		if w.Tick != 600 || len(w.Ants) == 0 {
			t.Fatalf("world %d at tick %d with %d ants", i, w.Tick, len(w.Ants))
		}
	}
}

// TestSchedulerKeepsWorldsInStep splits 7 worlds into 3 uneven groups and runs
// several short epochs. At every quiet point each world must be at the tick
// the epochs add up to, whatever order the groups were scheduled in.
func TestSchedulerKeepsWorldsInStep(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	worlds := make([]*sim.World, 7)
	for i := range worlds {
		worlds[i] = sim.NewWorld()
	}
	var stopped atomic.Bool
	s, err := newScheduler(d, worlds, 3, &stopped)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if len(s.groups) != 3 || len(s.groups[0].worlds) != 2 || len(s.groups[1].worlds) != 2 || len(s.groups[2].worlds) != 3 {
		t.Fatalf("unexpected split: %d groups", len(s.groups))
	}
	tick := 0
	for _, steps := range []int{100, 1, 150, 49} {
		ran, err := s.run(steps)
		if err != nil {
			t.Fatal(err)
		}
		tick += steps
		for k, r := range ran {
			if r != steps {
				t.Fatalf("group %d ran %d ticks, want %d", k, r, steps)
			}
		}
		for i, w := range worlds {
			if w.Tick != tick {
				t.Fatalf("world %d at tick %d, want %d", i, w.Tick, tick)
			}
		}
	}
	gpuIdle, cpuIdle := s.takeStats()
	t.Logf("GPU idle %.0f%%, workers idle %.0f%%", 100*gpuIdle, 100*cpuIdle)
}

// TestSchedulerStops sets the stop flag during a run: every group must drain
// at its next round, the worlds of a group staying in the same tick, and run
// must return the ticks done.
func TestSchedulerStops(t *testing.T) {
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	sim.LoadConfig("config.yml")
	sim.SavePath = t.TempDir() + "/colonies.json"
	worlds := make([]*sim.World, 5)
	for i := range worlds {
		worlds[i] = sim.NewWorld()
	}
	var stopped atomic.Bool
	s, err := newScheduler(d, worlds, 3, &stopped)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	go func() {
		time.Sleep(300 * time.Millisecond)
		stopped.Store(true)
	}()
	var ran []int
	waitReturn(t, func() { ran, err = s.run(1 << 30) })
	if err != nil {
		t.Fatal(err)
	}
	for k, g := range s.groups {
		if ran[k] == 0 {
			t.Fatalf("group %d ran no tick", k)
		}
		for _, w := range g.worlds {
			if w.Tick != ran[k] {
				t.Fatalf("group %d: a world at tick %d, group ran %d", k, w.Tick, ran[k])
			}
		}
	}
}

// TestRunWorldsHonoursTicks checks that the scheduler's prime and drain do not
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
	runWorlds(worlds, 7, 3, time.Hour, true, 3)
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
	waitReturn(t, func() { RunWorlds(4, 6000, 500, time.Hour, true, 3) })
	if _, err := os.Stat(sim.SavePath); err != nil {
		t.Fatalf("no save after the run: %v", err)
	}
}

// TestRunWorldsInterrupted sends SIGINT to an unbounded run, as Ctrl+C does:
// the scheduler must drain, save and return, for several worlds and for one.
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
		waitReturn(t, func() { RunWorlds(n, 0, 100000, time.Hour, true, 3) })
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
