// Headless evolution: runs the world as fast as the CPU allows, without any
// window, and saves the lineages so the game picks them up afterwards.
//
//	go build -o neatants-headless ./cmd/neatants-headless
//	./neatants-headless                 # until Ctrl+C
//	./neatants-headless -ticks 2000000  # a fixed number of ticks (per world)
//	./neatants-headless -worlds 1       # a single world (default: one per core)
package main

import (
	"flag"
	"fmt"
	"github.com/ThiraSoft/neatants/internal/headless"
	"github.com/ThiraSoft/neatants/internal/sim"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strings"
	"syscall"
	"time"
)

func main() {
	ticks := flag.Int("ticks", 0, "number of ticks to simulate (0 = until Ctrl+C)")
	every := flag.Duration("report", 10*time.Second, "interval between two reports")
	config := flag.String("config", "config.yml", "configuration file")
	worlds := flag.Int("worlds", runtime.NumCPU(), "worlds evolving in parallel (1 = a single world, brains spread over the cores)")
	epoch := flag.Int("migrate", 20000, "ticks between two champion migrations from one world to the next")
	gpu := flag.Bool("gpu", false, "think on the GPU (Vulkan), batching every world's brains")
	flag.Parse()
	given := false
	flag.Visit(func(f *flag.Flag) { given = given || f.Name == "worlds" })
	if *gpu && !given {
		*worlds = 48
	}

	// The simulation allocates little per tick: fewer, larger GC cycles.
	debug.SetGCPercent(400)
	sim.LoadConfig(*config)
	if *worlds > 1 || *gpu {
		headless.RunWorlds(*worlds, *ticks, max(500, *epoch), *every, *gpu)
		return
	}
	w := sim.NewWorld()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	start, last, lastTick := time.Now(), time.Now(), w.Tick
	fmt.Printf("[HEADLESS] starting · Ctrl+C to stop and save\n")
	for n := 0; *ticks == 0 || n < *ticks; n++ {
		if w.Update() {
			sim.SaveWorld(w)
		}
		w.Events = w.Events[:0]
		if n%500 != 0 {
			continue
		}
		select {
		case <-stop:
			sim.SaveWorld(w)
			fmt.Printf("\n[HEADLESS] stopped at tick %d, lineages saved to %s\n", w.Tick, sim.SavePath)
			return
		default:
		}
		if time.Since(last) >= *every {
			tps := float64(w.Tick-lastTick) / time.Since(last).Seconds()
			last, lastTick = time.Now(), w.Tick
			report(w, tps, time.Since(start))
		}
	}
	sim.SaveWorld(w)
	fmt.Printf("[HEADLESS] %d ticks simulated, lineages saved to %s\n", *ticks, sim.SavePath)
}

func report(w *sim.World, tps float64, elapsed time.Duration) {
	simulated := time.Duration(float64(w.Tick) / 60 * float64(time.Second))
	lineages := time.Duration(float64(w.Evolved) / 60 * float64(time.Second))
	fmt.Printf("\n[%s] tick %d · %.0f ticks/s (x%.0f) · wave %d · %s of game time simulated this session · lineages evolved for %.0fh · learning rewards %.0f %%\n",
		elapsed.Round(time.Second), w.Tick, tps, tps/60, w.Wave, simulated.Round(time.Minute), lineages.Hours(), 100*w.Shaping())
	mTop, mMean := 0.0, 0.0
	if len(w.MonHall) > 0 {
		mTop = w.MonHall[0].Fitness
	}
	for _, g := range w.MonRecent {
		mMean += g.Fitness
	}
	mMean /= float64(max(1, len(w.MonRecent)))
	if l := w.Learn; l.Deaths > 0 {
		per := 10000 / float64(max(1, l.Life))
		fmt.Printf("  %-11s %d deaths · %.2f deliveries and %.3f monsters per ant per 10k ticks of life · mean life %d ticks · starvation %.0f %% · killed by monsters %.0f %%\n",
			"Progress", l.Deaths, float64(l.Delivered)*per, float64(l.MonsterKills)*per, l.Life/l.Deaths, 100*float64(l.Starved)/float64(l.Deaths), 100*float64(l.ByMonsters)/float64(l.Deaths))
	}
	w.Learn = sim.LearnStats{}
	if f := w.Falls; f.Total() > 0 || f.Relocated > 0 {
		fmt.Printf("  %-11s %d collapses (monsters %d · raids %d · starvation %d) · mean age %d ticks · %d nests lost then relocated\n",
			"Colonies", f.Total(), f.Monsters, f.Raids, f.Starved, f.Age/max(1, f.Total()), f.Relocated)
	}
	w.Falls = sim.FallStats{}
	fmt.Printf("  %-11s threat %.2f · fitness top %6.2f avg %6.2f (ants hit and killed) · food at least %.0f px from the nests\n",
		"Cave", w.Threat, mTop, mMean, w.CurrentFoodGap())
	for _, c := range w.Colonies {
		if !c.Alive {
			fmt.Printf("  %-11s collapsed\n", c.Name)
			continue
		}
		var elems []string
		total := 0
		for _, n := range c.ElemCount {
			total += n
		}
		for e, n := range c.ElemCount {
			if n > 0 && total > 0 {
				elems = append(elems, fmt.Sprintf("%s %d%%", sim.ElemNames[e], 100*n/total))
			}
		}
		inst, cnt := 0.0, 0
		for _, a := range w.Ants {
			if a.Alive && a.Colony == c.ID {
				inst += a.Instinct
				cnt++
			}
		}
		spouse := ""
		if s := w.SpouseOf(c.ID); s >= 0 {
			spouse = " · married to " + w.Colonies[s].Name
		}
		// Best genome of the hall (aged a little at every entry) and mean of
		// the latest deaths kept for breeding.
		top, recent := 0.0, 0.0
		if len(c.Hall) > 0 {
			top = c.Hall[0].Fitness
		}
		for _, g := range c.Recent {
			recent += g.Fitness
		}
		recent /= float64(max(1, len(c.Recent)))
		fmt.Printf("  %-11s %3d ants · %2d eggs · stock %5.0f · %5d deliveries · %5d monsters · fitness top %6.2f avg %6.2f · instinct %.2f · %s%s\n",
			c.Name, c.Pop, c.Eggs, c.Food, c.Delivered, c.MonsterKills, top, recent, inst/float64(max(1, cnt)), strings.Join(elems, ", "), spouse)
	}
}
