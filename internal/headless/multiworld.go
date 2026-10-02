package headless

import (
	"fmt"
	"github.com/ThiraSoft/neatants/internal/sim"
	"github.com/ThiraSoft/neatants/neat"
	"os"
	"os/signal"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Parallel evolution on islands: several worlds start from the same save and
// evolve side by side, one goroutine (one core) each. Every epoch they pause
// together, the champion of each colony sails to the same colony of the next
// world (a ring), then they go on. The save merges the best of every world.
//
// All the worlds live in the same process on purpose: they share the NEAT
// innovation history, so a migrant's genes line up with its hosts' genes
// in crossover.

// RunWorlds evolves n worlds in parallel until ticks per world are done (0 = until
// interrupted), exchanging champions every epoch ticks.
func RunWorlds(n, ticks, epoch int, every time.Duration) {
	worlds := make([]*sim.World, n)
	for i := range worlds {
		worlds[i] = sim.NewWorld()
		worlds[i].SerialBrains = true
	}
	var stopped atomic.Bool
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		stopped.Store(true)
	}()

	fmt.Printf("[HEADLESS] %d worlds in parallel · migration every %d ticks · Ctrl+C to stop and save\n", n, epoch)
	start, last, lastSave := time.Now(), time.Now(), time.Now()
	done, lastDone := 0, 0
	for !stopped.Load() && (ticks == 0 || done < ticks) {
		steps := epoch
		if ticks > 0 {
			steps = min(steps, ticks-done)
		}
		ran := make([]int, n)
		var wg sync.WaitGroup
		for i, w := range worlds {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := 0; k < steps; k++ {
					w.Update()
					w.Events = w.Events[:0]
					ran[i]++
					if k%500 == 0 && stopped.Load() {
						return
					}
				}
			}()
		}
		wg.Wait()
		done += minInts(ran)
		migrate(worlds)

		if time.Since(lastSave) >= 30*time.Second {
			saveMerged(worlds)
			lastSave = time.Now()
		}
		if time.Since(last) >= every {
			tps := float64(done-lastDone) / time.Since(last).Seconds()
			last, lastDone = time.Now(), done
			reportWorlds(worlds, tps, time.Since(start))
		}
	}
	saveMerged(worlds)
	fmt.Printf("\n[HEADLESS] stopped after %d ticks per world (%d in total), lineages merged into %s\n", done, done*n, sim.SavePath)
}

func minInts(v []int) int {
	m := v[0]
	for _, x := range v[1:] {
		m = min(m, x)
	}
	return m
}

// migrate sends a copy of each colony's champion, and of the cave's, to the
// next world. A migrant keeps its fitness: it only stays in its new hall if
// it beats the locals.
func migrate(worlds []*sim.World) {
	if len(worlds) < 2 {
		return
	}
	type migrant struct {
		col int // colony index, -1 for the cave
		g   *neat.Genome
	}
	out := make([][]migrant, len(worlds))
	for i, w := range worlds {
		for ci, c := range w.Colonies {
			if len(c.Hall) > 0 {
				out[i] = append(out[i], migrant{ci, c.Hall[0].Copy()})
			}
		}
		if len(w.MonHall) > 0 {
			out[i] = append(out[i], migrant{-1, w.MonHall[0].Copy()})
		}
	}
	for i, ms := range out {
		dst := worlds[(i+1)%len(worlds)]
		for _, m := range ms {
			if m.col < 0 {
				adopt(&dst.MonHall, m.g, dst.MonSpecies.Threshold())
			} else if m.col < len(dst.Colonies) {
				c := dst.Colonies[m.col]
				adopt(&c.Hall, m.g, c.Species.Threshold())
			}
		}
	}
}

// adopt inserts g in hall unless its lineage is already there (both worlds
// may hold the same champion since they started from the same save).
func adopt(hall *[]*neat.Genome, g *neat.Genome, dist float64) {
	id := g.Lineage()
	for _, h := range *hall {
		if h.Lineage() == id {
			return
		}
	}
	if sim.Advanced() {
		sim.InsertHallSpeciated(hall, g, dist)
	} else {
		sim.InsertHallOfFame(hall, g)
	}
}

// saveMerged writes one save holding, for each colony, the best genomes of
// all the worlds (one per lineage, at most speciesCap per species), so the
// game and the next session pick up the best of every island.
func saveMerged(worlds []*sim.World) {
	best := func(pick func(w *sim.World) []*neat.Genome, dist float64) []*neat.Genome {
		var all []*neat.Genome
		for _, w := range worlds {
			all = append(all, pick(w)...)
		}
		sort.SliceStable(all, func(a, b int) bool { return all[a].Fitness > all[b].Fitness })
		seen := map[int]bool{}
		var hall []*neat.Genome
		for _, g := range all {
			if len(hall) >= sim.Cfg.HallOfFameSize {
				break
			}
			if seen[g.Lineage()] {
				continue
			}
			same := 0
			for _, h := range hall {
				if neat.Compatibility(g, h) < dist {
					same++
				}
			}
			if sim.Advanced() && same >= sim.SpeciesCap {
				continue
			}
			seen[g.Lineage()] = true
			hall = append(hall, g)
		}
		return hall
	}
	// The world with the strongest champion lends its species registry.
	top := func(pick func(w *sim.World) []*neat.Genome) *sim.World {
		var tw *sim.World
		bf := -1.0
		for _, w := range worlds {
			if h := pick(w); len(h) > 0 && h[0].Fitness > bf {
				tw, bf = w, h[0].Fitness
			}
		}
		return tw
	}

	w0 := worlds[0]
	s := sim.SaveData{Tick: w0.Tick, Threat: w0.Threat, FoodGap: w0.FoodGap}
	for _, w := range worlds {
		s.Evolved = max(s.Evolved, w.Evolved)
	}
	monHall := func(w *sim.World) []*neat.Genome { return w.MonHall }
	if tw := top(monHall); tw != nil {
		s.Monsters = best(monHall, tw.MonSpecies.Threshold())
		s.MonSpecies = &tw.MonSpecies
	}
	// Fallen colonies are founded again: some worlds may hold more of them.
	nc := 0
	for _, w := range worlds {
		nc = max(nc, len(w.Colonies))
	}
	for ci := range nc {
		hall := func(w *sim.World) []*neat.Genome {
			if ci < len(w.Colonies) {
				return w.Colonies[ci].Hall
			}
			return nil
		}
		tw := top(hall)
		if tw == nil {
			continue
		}
		c := tw.Colonies[ci]
		s.Colonies = append(s.Colonies, sim.SavedColony{Name: c.Name, Hall: best(hall, c.Species.Threshold()), Species: &c.Species})
	}
	sim.WriteSave(s)
}

// reportWorlds prints one line per world: the learning bench (deliveries per
// ant and 10k ticks of life) and the colonies still standing.
func reportWorlds(worlds []*sim.World, tps float64, elapsed time.Duration) {
	fmt.Printf("\n[%s] %d ticks/s per world · %.0f ticks/s in total (x%.0f)\n",
		elapsed.Round(time.Second), int(tps), tps*float64(len(worlds)), tps*float64(len(worlds))/60)
	var all sim.LearnStats
	for i, w := range worlds {
		l := w.Learn
		w.Learn = sim.LearnStats{}
		w.Falls = sim.FallStats{}
		alive, pop := 0, 0
		for _, c := range w.Colonies {
			if c.Alive {
				alive++
				pop += c.Pop
			}
		}
		if l.Deaths == 0 {
			fmt.Printf("  world %-2d tick %d · %d colonies · %d ants · no deaths\n", i+1, w.Tick, alive, pop)
			continue
		}
		per := 10000 / float64(max(1, l.Life))
		all.Delivered += l.Delivered
		all.Life += l.Life
		fmt.Printf("  world %-2d tick %d · %d colonies · %3d ants · %.2f deliveries and %.3f monsters /ant/10k ticks · life %d · starvation %.0f %% · monsters %.0f %%\n",
			i+1, w.Tick, alive, pop, float64(l.Delivered)*per, float64(l.MonsterKills)*per, l.Life/l.Deaths,
			100*float64(l.Starved)/float64(l.Deaths), 100*float64(l.ByMonsters)/float64(l.Deaths))
	}
	if all.Life > 0 {
		fmt.Printf("  %-8s %.2f deliveries /ant/10k ticks across all worlds\n", "Archipel", float64(all.Delivered)*10000/float64(all.Life))
	}
}
