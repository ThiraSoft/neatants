package sim

import (
	"encoding/json"
	"fmt"
	"github.com/ThiraSoft/neatants/neat"
	"os"
	"path/filepath"
)

const (
	SavePath     = "saves/colonies.json"
	prevSavePath = "saves/colonies.prev.json"
)

type SaveData struct {
	Tick       int            `json:"tick"`
	Evolved    int            `json:"evolved"`
	Threat     float64        `json:"threat,omitempty"`
	FoodGap    float64        `json:"food_gap,omitempty"`
	Monsters   []*neat.Genome `json:"monsters,omitempty"`
	MonSpecies *Speciation    `json:"mon_species,omitempty"`
	Colonies   []SavedColony  `json:"colonies"`
}

type SavedColony struct {
	Name    string         `json:"name"`
	Hall    []*neat.Genome `json:"hall"`
	Species *Speciation    `json:"species,omitempty"`
}

// Restored is what a save gives back to a new world.
type Restored struct {
	Pools    [][]*neat.Genome // gene pool of each colony (may be empty)
	Evolved  int              // ticks these lineages have already evolved
	Threat   float64
	FoodGap  float64
	Monsters []*neat.Genome // the cave's gene pool
	// Species registries, aligned with Pools (nil when absent or unusable).
	Species    []*Speciation
	MonSpecies *Speciation
}

// usableSpecies keeps a saved registry only if every representative fits
// the current brains, and registers their innovations.
func usableSpecies(s *Speciation, inputs, outputs int) *Speciation {
	if s == nil {
		return nil
	}
	var reps []*neat.Genome
	for _, sp := range s.List {
		if sp == nil || sp.Rep == nil || sp.Rep.NumInputs != inputs || sp.Rep.NumOutputs != outputs {
			return nil
		}
		reps = append(reps, sp.Rep)
	}
	neat.SyncInnovations(reps)
	return s
}

// LoadSave reads the save, keeping only genomes that fit the current brains.
func LoadSave() Restored {
	data, err := os.ReadFile(SavePath)
	if err != nil {
		return Restored{}
	}
	var s SaveData
	if err := json.Unmarshal(data, &s); err != nil {
		fmt.Printf("[LOAD] unreadable save: %v\n", err)
		return Restored{}
	}
	var pools [][]*neat.Genome
	var species []*Speciation
	genomes := 0
	for _, c := range s.Colonies {
		var ok []*neat.Genome
		for _, g := range c.Hall {
			if g != nil && g.NumInputs == AntInputs && g.NumOutputs == AntOutputs {
				ok = append(ok, g)
			}
		}
		neat.SyncInnovations(ok)
		pools = append(pools, ok)
		species = append(species, usableSpecies(c.Species, AntInputs, AntOutputs))
		genomes += len(ok)
	}
	if genomes == 0 {
		s.Evolved = 0 // nothing usable: the lineages start over
	}
	var mons []*neat.Genome
	for _, g := range s.Monsters {
		if g != nil && g.NumInputs == MonInputs && g.NumOutputs == MonOutputs {
			mons = append(mons, g)
		}
	}
	neat.SyncInnovations(mons)
	fmt.Printf("[LOAD] %d lineages restored from %s (%d evolution ticks, %d monster brains, threat %.2f)\n",
		len(pools), SavePath, s.Evolved, len(mons), s.Threat)
	return Restored{Pools: pools, Evolved: s.Evolved, Threat: s.Threat, FoodGap: s.FoodGap, Monsters: mons,
		Species: species, MonSpecies: usableSpecies(s.MonSpecies, MonInputs, MonOutputs)}
}

// SaveWorld writes the gene pool of every colony that has one, living
// colonies first. A fallen colony still keeps its hall of fame, so a total
// wipe-out never erases the lineages; and an empty pool is never written
// over an existing save. The previous file is kept as colonies.prev.json.
func SaveWorld(w *World) {
	s := SaveData{Tick: w.Tick, Evolved: w.Evolved, Threat: w.Threat, FoodGap: w.FoodGap, Monsters: w.MonHall, MonSpecies: &w.MonSpecies}
	var fallen []SavedColony
	for _, c := range w.Colonies {
		if len(c.Hall) == 0 {
			continue
		}
		sc := SavedColony{Name: c.Name, Hall: c.Hall, Species: &c.Species}
		if c.Alive {
			s.Colonies = append(s.Colonies, sc)
		} else {
			fallen = append(fallen, sc)
		}
	}
	s.Colonies = append(s.Colonies, fallen...)
	WriteSave(s)
}

// writeSave writes s atomically, keeping the previous file aside. A save
// without any colony is never written.
func WriteSave(s SaveData) {
	if len(s.Colonies) == 0 {
		return
	}
	os.MkdirAll(filepath.Dir(SavePath), 0755)
	data, err := json.Marshal(s)
	if err != nil {
		return
	}
	tmp := SavePath + ".tmp"
	if os.WriteFile(tmp, data, 0644) != nil {
		return
	}
	if _, err := os.Stat(SavePath); err == nil {
		os.Rename(SavePath, prevSavePath)
	}
	os.Rename(tmp, SavePath)
}
