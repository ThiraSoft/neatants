package evo

import (
	"math"
	"testing"

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
