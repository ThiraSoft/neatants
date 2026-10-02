package evo

import (
	"math"
	"slices"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// toy rewards genomes for having many enabled connections, so that a
// working loop must grow them.
func toy(g *neat.Genome) float64 {
	n := 0
	for _, c := range g.Conns {
		if c.Enabled {
			n++
		}
	}
	return float64(n)
}

func scores(p *Population, f func(*neat.Genome) float64) []float64 {
	s := make([]float64, len(p.Genomes))
	for i, g := range p.Genomes {
		s[i] = f(g)
	}
	return s
}

func TestPopulationSizeIsKept(t *testing.T) {
	p := New(DefaultConfig(), 8)
	for range 20 {
		p.Next(scores(p, toy))
		if len(p.Genomes) != DefaultConfig().Pop {
			t.Fatalf("gen %d: %d genomes", p.Gen, len(p.Genomes))
		}
	}
}

func TestToyImproves(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pop = 200
	p := New(cfg, 8)
	p.Next(scores(p, toy))
	first := toy(p.Best())
	for range 60 {
		p.Next(scores(p, toy))
	}
	if toy(p.Best()) <= first {
		t.Fatalf("best stayed at %g", first)
	}
}

func TestFlatAndInfiniteFitness(t *testing.T) {
	p := New(DefaultConfig(), 8)
	flat := make([]float64, len(p.Genomes))
	p.Next(flat)
	bad := make([]float64, len(p.Genomes))
	for i := range bad {
		bad[i] = -1
		if i%3 == 0 {
			bad[i] = math.Inf(-1)
		}
	}
	p.Next(bad)
	if len(p.Genomes) != DefaultConfig().Pop {
		t.Fatal(len(p.Genomes))
	}
	for _, g := range p.Genomes {
		if math.IsNaN(g.Fitness) {
			t.Fatal("NaN fitness")
		}
	}
}

func TestChampionSurvivesUnchanged(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pop = 100
	p := New(cfg, 8)
	p.Next(scores(p, toy))
	best := p.Best()
	found := false
	for _, g := range p.Genomes {
		if g.Lineage() == best.Lineage() && len(g.Conns) == len(best.Conns) {
			found = true
		}
	}
	if !found {
		t.Fatal("the champion was not carried over")
	}
}

// The threshold must follow the population as it converges: relatives soon
// differ by a few hundredths in Compatibility, far under the 2.0 the
// threshold starts at, and the species count must still reach the band.
func TestSpeciesStayInBand(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pop = 200
	p := New(cfg, 32)
	in := 0
	var counts []int
	for gen := range 100 {
		p.Next(scores(p, toy))
		counts = append(counts, len(p.Species))
		if gen >= 50 && len(p.Species) >= cfg.MinSpecies && len(p.Species) <= cfg.MaxSpecies {
			in++
		}
	}
	t.Logf("species per generation: %v, threshold %.4f", counts, p.Threshold)
	if in < 40 {
		t.Fatalf("only %d of the last 50 generations have %d to %d species", in, cfg.MinSpecies, cfg.MaxSpecies)
	}
}

// speciateSequential is the one-genome-at-a-time speciation the parallel one
// must reproduce: each genome joins the first species, in registry order,
// whose representative is close enough, the species created earlier in the
// same pass included, or founds a new one.
func speciateSequential(species []*Species, genomes []*neat.Genome, threshold float64) [][]int {
	var reps []*neat.Genome
	for _, sp := range species {
		reps = append(reps, sp.Rep)
	}
	members := make([][]int, len(reps))
	for _, g := range genomes {
		home := -1
		for k, r := range reps {
			if neat.Compatibility(g, r) < threshold {
				home = k
				break
			}
		}
		if home < 0 {
			reps = append(reps, g)
			members = append(members, nil)
			home = len(reps) - 1
		}
		members[home] = append(members[home], g.ID)
	}
	var out [][]int
	for _, m := range members {
		if len(m) > 0 {
			out = append(out, m)
		}
	}
	return out
}

// The parallel speciation must give the same species, with the same members
// in the same order, as the sequential pass, both when most genomes find an
// existing species and when many found new ones in the same generation.
func TestParallelSpeciationMatchesSequential(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pop = 300
	p := New(cfg, 16)
	for gen := range 30 {
		fit := scores(p, toy)
		p.record(fit)
		for _, th := range []float64{p.Threshold, p.Threshold / 4} {
			saved := p.Threshold
			p.Threshold = th
			species := make([]*Species, len(p.Species))
			for i, sp := range p.Species {
				c := *sp
				species[i] = &c
			}
			want := speciateSequential(p.Species, p.Genomes, th)
			q := *p
			q.Species = species
			q.speciate()
			var got [][]int
			for _, sp := range q.Species {
				var ids []int
				for _, g := range sp.Members {
					ids = append(ids, g.ID)
				}
				got = append(got, ids)
			}
			if len(got) != len(want) {
				t.Fatalf("gen %d threshold %g: %d species, sequential %d", gen, th, len(got), len(want))
			}
			for k := range want {
				if !slices.Equal(got[k], want[k]) {
					t.Fatalf("gen %d threshold %g: species %d has %v, sequential %v", gen, th, k, got[k], want[k])
				}
			}
			p.Threshold = saved
		}
		// The real step, so that the next generation starts from it.
		for _, g := range p.Genomes {
			g.Evals = 0
		}
		p.Next(fit)
	}
}

// Every genome of a population with state banks, children included, keeps
// the shape of the first generation.
func TestNewShapeKeepsTheShape(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Pop = 100
	s := model.Shape{Dim: 8, Banks: 3}
	p := NewShape(cfg, s)
	for range 10 {
		for _, g := range p.Genomes {
			if got := model.ShapeOf(g, 8); got != s {
				t.Fatalf("gen %d: genome %d has shape %+v", p.Gen, g.ID, got)
			}
		}
		p.Next(scores(p, toy))
	}
}
