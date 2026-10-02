package headless

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/gpubrain"
	"github.com/ThiraSoft/neatants/internal/sim"
)

// slotsPerWorld sizes the arena so a normal run never grows it: growing
// allocates on the card under the mutex while every worker waits.
const slotsPerWorld = 300

// stepMode tells the workers what to do with a world in the current round of
// its group.
type stepMode int

const (
	modePrime stepMode = iota // Sense and Add only: the first half of the epoch's first tick
	modeStep                  // copy outputs, Act, Sense, Add
	modeDrain                 // copy outputs and Act only: the last tick of the epoch
)

// group is a set of worlds whose brains think in one GPU batch. All its
// worlds are in the same tick: a round of the group is one tick of each.
type group struct {
	worlds []*sim.World
	batch  *gpubrain.Batch
	idx    [][]int // per world, the ticket of each of its Thoughts in the batch, reused between rounds
	last   int     // requests of the previous round, to size the next

	// Scheduling state. The owner (the GPU goroutine, or run before it
	// starts) writes mode and acts only while no worker holds a world of the
	// group, and hands the worlds over through the ready channel, which orders
	// the writes before the workers' reads. pending counts the worlds the
	// workers still have to step in this round.
	mode    stepMode
	acts    int // Acts done or under way in this epoch: the ticks the group has run
	pending atomic.Int32
}

func newGroup(d *vk.Device, worlds []*sim.World) (*group, error) {
	for _, w := range worlds {
		w.SerialBrains = true
	}
	inStride := max(sim.AntInputs, sim.MonInputs)
	outStride := max(sim.AntOutputs, sim.MonOutputs)
	b, err := gpubrain.New(d, inStride, outStride, max(1, sim.Cfg.ThinkEvery)+1, slotsPerWorld*len(worlds))
	if err != nil {
		return nil, err
	}
	return &group{worlds: worlds, batch: b, idx: make([][]int, len(worlds))}, nil
}

// open opens the batch for the next round. The capacity covers the last
// round with some room; the batch doubles it anyway after an overflow.
func (g *group) open() {
	g.last = 0
	for _, idx := range g.idx {
		g.last += len(idx)
	}
	g.batch.Open(g.last*5/4 + 64*len(g.worlds))
}

// stepWorld does what mode asks for world i: the outputs of the last round
// into its Thoughts, Act, then Sense and Add of every Thought to the batch.
// Workers call it concurrently for different worlds of a group. The outputs
// of the last round stay valid while other workers Add to the next one.
func (g *group) stepWorld(i int) {
	w := g.worlds[i]
	if g.mode != modePrime {
		for j, t := range w.Thoughts {
			o := g.batch.Out(g.idx[i][j])
			for k := range t.Out {
				t.Out[k] = float64(o[k])
			}
		}
		w.Act() // the per-world save flag is ignored: RunWorlds saves the merged worlds
		w.Events = w.Events[:0]
	}
	if g.mode == modeDrain {
		return
	}
	w.Sense()
	idx := g.idx[i][:0]
	for _, t := range w.Thoughts {
		idx = append(idx, g.batch.Add(t.Net, t.In))
	}
	g.idx[i] = idx
}

func (g *group) close() { g.batch.Close() }

// scheduler keeps the cores stepping worlds while the card thinks for other
// groups. Two kinds of goroutines, all started and joined by run:
//
//   - workers take one world at a time from the ready queue and step it
//     (stepWorld). The worker that steps the last world of a group hands the
//     group to the GPU goroutine through the filled queue, or reports it
//     drained.
//   - one GPU goroutine takes the filled groups in order, Starts and Finishes
//     each, opens its next round and puts all its worlds on the ready queue.
//
// Invariants:
//   - Only the GPU goroutine calls Start and Finish, so at most one dispatch
//     is in flight on the device, as vk.Device.Start requires, and a group's
//     Open comes after its Finish.
//   - A world is on the ready queue, or held by one worker, or waiting in a
//     group the GPU goroutine owns, never two at once: the ready queue is as
//     big as the worlds, so no send on it ever blocks.
//   - A group is dispatched again only after all its worlds added, and its
//     workers copy the outputs of round r before adding to round r+1, so the
//     outputs of round r are never read after round r+2 opens.
//   - Every group runs the same number of acts in an epoch, except when the
//     run is stopped, which each group notices at its next round. So all the
//     worlds of a group are always in the same tick.
type scheduler struct {
	groups  []*group
	stopped *atomic.Bool
	workers int

	// Idle time, in nanoseconds, since the last takeStats: how long the GPU
	// goroutine waited for a filled group (the card was idle), how long the
	// workers together waited for a ready world (cores idle), and the time
	// spent in run.
	gpuWait, workerWait, wall atomic.Int64
}

// newScheduler splits worlds into at most nGroups groups, as even as possible.
func newScheduler(d *vk.Device, worlds []*sim.World, nGroups int, stopped *atomic.Bool) (*scheduler, error) {
	nGroups = max(1, min(nGroups, len(worlds)))
	s := &scheduler{stopped: stopped, workers: min(runtime.NumCPU(), len(worlds))}
	from := 0
	for k := range nGroups {
		to := from + (len(worlds)-from)/(nGroups-k)
		g, err := newGroup(d, worlds[from:to])
		if err != nil {
			s.close()
			return nil, err
		}
		s.groups = append(s.groups, g)
		from = to
	}
	return s, nil
}

// run advances every world by steps ticks, or fewer if the run is stopped,
// and returns the ticks each group ran, in group order. Afterwards all the
// worlds are quiet, between two ticks, and run can be called again: it
// primes them with a Sense and an Add first. On a GPU error it stops the
// workers and returns it, the worlds being at arbitrary ticks but not in use.
func (s *scheduler) run(steps int) ([]int, error) {
	begin := time.Now()
	total := 0
	for _, g := range s.groups {
		total += len(g.worlds)
	}
	ready := make(chan job, total)
	filled := make(chan *group, len(s.groups))
	drained := make(chan struct{}, len(s.groups))
	errc := make(chan error, 1)
	quit := make(chan struct{})

	var workers, gpu sync.WaitGroup
	for range s.workers {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for {
				t := time.Now()
				j, ok := <-ready
				s.workerWait.Add(int64(time.Since(t)))
				if !ok {
					return
				}
				j.g.stepWorld(j.i)
				if j.g.pending.Add(-1) > 0 {
					continue
				}
				if j.g.mode == modeDrain {
					drained <- struct{}{}
				} else {
					filled <- j.g
				}
			}
		}()
	}
	gpu.Add(1)
	go func() {
		defer gpu.Done()
		for {
			t := time.Now()
			var g *group
			select {
			case g = <-filled:
			case <-quit:
				return
			}
			s.gpuWait.Add(int64(time.Since(t)))
			err := g.batch.Start()
			if err == nil {
				err = g.batch.Finish()
			}
			if err != nil {
				errc <- err
				return
			}
			// The next round acts once more. The last one of the epoch, or
			// the first after a stop, only acts.
			g.acts++
			g.mode = modeStep
			if g.acts >= steps || s.stopped.Load() {
				g.mode = modeDrain
			} else {
				g.open()
			}
			g.release(ready)
		}
	}()

	// Open every batch before any dispatch can start, so no buffer is
	// allocated on the device while another group's dispatch is in flight.
	for _, g := range s.groups {
		g.acts, g.mode = 0, modePrime
		g.open()
	}
	for _, g := range s.groups {
		g.release(ready)
	}
	var err error
	for left := len(s.groups); left > 0 && err == nil; {
		select {
		case <-drained:
			left--
		case err = <-errc:
		}
	}
	// On the way out nothing sends on ready any more: the GPU goroutine has
	// returned, by quit or by its error.
	close(quit)
	gpu.Wait()
	close(ready)
	workers.Wait()
	s.wall.Add(int64(time.Since(begin)))
	ran := make([]int, len(s.groups))
	for i, g := range s.groups {
		ran[i] = g.acts
	}
	return ran, err
}

type job struct {
	g *group
	i int
}

// release puts every world of g on the ready queue.
func (g *group) release(ready chan<- job) {
	g.pending.Store(int32(len(g.worlds)))
	for i := range g.worlds {
		ready <- job{g, i}
	}
}

// takeStats returns the share of the time the card waited for a group to
// fill, and the share the workers waited for a ready world, since the last
// call.
func (s *scheduler) takeStats() (gpuIdle, cpuIdle float64) {
	wall := float64(max(1, s.wall.Swap(0)))
	gpuIdle = float64(s.gpuWait.Swap(0)) / wall
	cpuIdle = float64(s.workerWait.Swap(0)) / (wall * float64(max(1, s.workers)))
	return
}

func (s *scheduler) close() {
	for _, g := range s.groups {
		g.close()
	}
}
