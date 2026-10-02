package headless

import (
	"runtime"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/gpubrain"
	"github.com/ThiraSoft/neatants/internal/sim"
)

// group is a set of worlds whose brains think in one GPU batch.
type group struct {
	worlds []*sim.World
	batch  *gpubrain.Batch
	first  []int // first request of each world in the batch
}

func newGroup(d *vk.Device, worlds []*sim.World) (*group, error) {
	for _, w := range worlds {
		w.SerialBrains = true
	}
	inStride := max(sim.AntInputs, sim.MonInputs)
	outStride := max(sim.AntOutputs, sim.MonOutputs)
	b, err := gpubrain.New(d, inStride, outStride, max(1, sim.Cfg.ThinkEvery)+1, 256*len(worlds))
	if err != nil {
		return nil, err
	}
	return &group{worlds: worlds, batch: b, first: make([]int, len(worlds)+1)}, nil
}

// sense runs Sense on every world and fills the batch.
func (g *group) sense() {
	forEach(len(g.worlds), func(i int) { g.worlds[i].Sense() })
	for i, w := range g.worlds {
		g.first[i+1] = g.first[i] + len(w.Thoughts)
	}
	g.batch.Begin(g.first[len(g.worlds)])
	forEach(len(g.worlds), func(i int) {
		for j, t := range g.worlds[i].Thoughts {
			g.batch.Set(g.first[i]+j, t.Net, t.In)
		}
	})
}

func (g *group) start() error  { return g.batch.Start() }
func (g *group) finish() error { return g.batch.Finish() }

// act copies the outputs back and runs Act on every world.
func (g *group) act() {
	forEach(len(g.worlds), func(i int) {
		w := g.worlds[i]
		for j, t := range w.Thoughts {
			o := g.batch.Out(g.first[i] + j)
			for k := range t.Out {
				t.Out[k] = float64(o[k])
			}
		}
		w.Act() // the per-world save flag is ignored: RunWorlds saves the merged worlds
		w.Events = w.Events[:0]
	})
}

func (g *group) close() { g.batch.Close() }

// pipeline thinks for one half of the worlds on the card while the CPU
// steps the other half, so neither waits for the other more than the
// difference between the two. Only one dispatch is ever in flight, as
// vk.Device.Start requires.
type pipeline struct {
	a, b *group
	// cpuWait is the time the CPU spent waiting for the card. cpuStep is the
	// time it spent stepping a half (Act and Sense): an upper bound of the
	// card's busy time per half, since the card may finish earlier.
	cpuWait, cpuStep time.Duration
}

func newPipeline(d *vk.Device, worlds []*sim.World) (*pipeline, error) {
	h := (len(worlds) + 1) / 2
	a, err := newGroup(d, worlds[:h])
	if err != nil {
		return nil, err
	}
	b, err := newGroup(d, worlds[h:])
	if err != nil {
		a.close()
		return nil, err
	}
	return &pipeline{a: a, b: b}, nil
}

// prime runs the first Sense of every world and sets the card on A.
func (p *pipeline) prime() error {
	p.a.sense()
	if err := p.a.start(); err != nil {
		return err
	}
	p.b.sense()
	return nil
}

// step advances every world by one tick.
func (p *pipeline) step() error {
	if err := p.half(p.a, p.b); err != nil {
		return err
	}
	return p.half(p.b, p.a)
}

// half collects done's thoughts, sets the card on next, then steps done.
func (p *pipeline) half(done, next *group) error {
	t := time.Now()
	if err := done.finish(); err != nil {
		return err
	}
	p.cpuWait += time.Since(t)
	if err := next.start(); err != nil {
		return err
	}
	t = time.Now()
	done.act()
	done.sense()
	p.cpuStep += time.Since(t)
	return nil
}

// drain finishes the dispatch in flight and the pending ticks, so the worlds
// can be saved. They are then between ticks: prime resumes them.
func (p *pipeline) drain() error {
	if err := p.a.finish(); err != nil {
		return err
	}
	p.a.act()
	if err := p.b.start(); err != nil {
		return err
	}
	if err := p.b.finish(); err != nil {
		return err
	}
	p.b.act()
	return nil
}

func (p *pipeline) close() {
	p.a.close()
	p.b.close()
}

// forEach runs f(0..n-1) on at most NumCPU goroutines.
func forEach(n int, f func(i int)) {
	workers := min(n, runtime.NumCPU())
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := range n {
		next <- i
	}
	close(next)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				f(i)
			}
		}()
	}
	wg.Wait()
}
