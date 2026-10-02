package sim

import (
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"testing"
)

func TestSimulationStats(t *testing.T) {
	LoadConfig("config.yml")
	w := NewWorld()
	counts := map[EventKind]int{}
	roles := map[Role]int{}
	for i := 0; i < 6000; i++ {
		w.Update()
		for _, e := range w.Events {
			counts[e.Kind]++
		}
		w.Events = w.Events[:0]
		if i%1000 == 0 {
			for _, a := range w.Ants {
				if a.Alive {
					roles[a.Role]++
				}
			}
		}
	}
	fmt.Println("events:", counts)
	fmt.Println("roles:", roles)
	for _, c := range w.Colonies {
		sum, n, mem, memUsed := 0.0, 0, 0, 0
		for _, a := range w.Ants {
			if a.Alive && a.Colony == c.ID {
				sum += a.Instinct
				n++
				for i, nd := range a.Genome.Nodes {
					if nd.Type == neat.Memory {
						mem++
						if a.Net.Cell(i) != 0 {
							memUsed++
						}
					}
				}
			}
		}
		fmt.Printf("instinct %.2f memories/ant %.2f (active %d) ", sum/float64(max(n, 1)), float64(mem)/float64(max(n, 1)), memUsed)
		fmt.Printf("%s alive=%v pop=%d food=%.1f hp=%.0f deliv=%d kills=%d mk=%d elems=%v\n", c.Name, c.Alive, c.Pop, c.Food, c.HP, c.Delivered, c.Kills, c.MonsterKills, c.ElemCount)
	}
	fmt.Println("wave", w.Wave, "monsters alive", w.WaveAlive, "killed", w.TotalKilled)
}
