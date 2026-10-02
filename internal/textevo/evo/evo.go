// Package evo is a generational NEAT population: genomes are scored all at
// once, sorted into species, and replaced by the next generation.
package evo

import (
	"math"
	"math/rand"
	"sort"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/neat"
)

// Config holds the evolution parameters.
type Config struct {
	Pop                    int     // population size
	MinSpecies, MaxSpecies int     // the threshold drifts to stay in this range
	Stagnation             int     // generations without progress before a species stops breeding
	Survival               float64 // share of a species that may be a parent
	MutateOnly             float64 // share of children made by mutation alone
	Interspecies           float64 // share of crossovers with a mate of another species
	ChampionMin            int     // species at least this big keep their champion unchanged
}

// DefaultConfig returns the values used by the text evolution loop.
func DefaultConfig() Config {
	return Config{
		Pop:        1000,
		MinSpecies: 5, MaxSpecies: 15,
		Stagnation:   15,
		Survival:     0.2,
		MutateOnly:   0.25,
		Interspecies: 0.001,
		ChampionMin:  5,
	}
}

// Species is a group of compatible genomes, represented by one of them.
type Species struct {
	ID       int
	Rep      *neat.Genome
	Members  []*neat.Genome
	Best     float64
	Improved int // generation of the last improvement
}

// Population is the current generation and its species registry.
type Population struct {
	Genomes   []*neat.Genome
	Species   []*Species
	Gen       int
	Threshold float64

	cfg      Config
	dim      int
	nextID   int
	nextSpec int
	best     *neat.Genome
}

// New creates a random first generation of genomes for dim-wide vectors.
func New(cfg Config, dim int) *Population {
	p := &Population{cfg: cfg, dim: dim, nextID: cfg.Pop + 1, Threshold: 2.0}
	for i := 0; i < cfg.Pop; i++ {
		p.Genomes = append(p.Genomes, model.NewGenome(i+1, dim))
	}
	return p
}

// Best returns the genome with the highest recorded fitness of the generation
// just scored. Next stores it before replacing Genomes.
func (p *Population) Best() *neat.Genome { return p.best }

// Next takes the fitness of each genome of p.Genomes (higher is better,
// -Inf allowed), records it, and replaces p.Genomes with the next generation.
func (p *Population) Next(fitness []float64) {
	p.record(fitness)
	p.speciate()
	shares := p.allocate()
	p.Genomes = p.breed(shares)
	// Representatives are drawn among this generation's members, so that the
	// next speciation compares the children to genomes that still exist.
	for _, sp := range p.Species {
		sp.Rep = sp.Members[rand.Intn(len(sp.Members))]
	}
	p.Gen++
}

// record stores the fitness of each genome. Infinite and NaN scores become
// finite so that the allocation below never meets a NaN: -Inf is put just
// under the worst finite score.
func (p *Population) record(fitness []float64) {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, f := range fitness {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		lo, hi = math.Min(lo, f), math.Max(hi, f)
	}
	floor, ceil := lo-10, hi+10
	if math.IsInf(lo, 1) {
		floor, ceil = -1e9, -1e9
	}
	p.best = nil
	for i, g := range p.Genomes {
		f := fitness[i]
		switch {
		case math.IsNaN(f) || math.IsInf(f, -1):
			f = floor
		case math.IsInf(f, 1):
			f = ceil
		}
		if g.Evals > 0 {
			// An unchanged copy was scored before: average the lives.
			n := float64(g.Evals)
			g.Fitness = (g.Fitness*n + f) / (n + 1)
		} else {
			g.Fitness = f
		}
		g.Evals++
		if p.best == nil || g.Fitness > p.best.Fitness {
			p.best = g
		}
	}
}

// speciate sorts the genomes into the species of the previous generation,
// creating species for those that fit none, as in the steady-state version
// of internal/sim but over a whole generation.
func (p *Population) speciate() {
	for _, sp := range p.Species {
		sp.Members = sp.Members[:0]
	}
	for _, g := range p.Genomes {
		var home *Species
		for _, sp := range p.Species {
			if neat.Compatibility(g, sp.Rep) < p.Threshold {
				home = sp
				break
			}
		}
		if home == nil {
			p.nextSpec++
			home = &Species{ID: p.nextSpec, Rep: g, Best: g.Fitness, Improved: p.Gen}
			p.Species = append(p.Species, home)
		}
		home.Members = append(home.Members, g)
	}
	alive := p.Species[:0]
	for _, sp := range p.Species {
		if len(sp.Members) == 0 {
			continue
		}
		for _, g := range sp.Members {
			if g.Fitness > sp.Best {
				sp.Best, sp.Improved = g.Fitness, p.Gen
			}
		}
		alive = append(alive, sp)
	}
	p.Species = alive
	// Relatives in a converged population differ by a few hundredths (one
	// disjoint gene in two thousand is 5e-4), so the threshold may go that
	// low, and it moves by a fifth a generation so that it gets there from
	// its start at 2.0 in about twenty generations rather than a hundred.
	switch {
	case len(p.Species) < p.cfg.MinSpecies:
		p.Threshold = math.Max(1e-4, p.Threshold*0.8)
	case len(p.Species) > p.cfg.MaxSpecies:
		p.Threshold = math.Min(6, p.Threshold*1.2)
	}
}

// allocate gives each species its number of children, proportional to the
// mean adjusted fitness of its members. Species stagnant for too long get
// nothing, except the two with the best scores.
func (p *Population) allocate() []int {
	n := len(p.Species)
	lo := math.Inf(1)
	for _, g := range p.Genomes {
		lo = math.Min(lo, g.Fitness)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return p.Species[order[a]].Best > p.Species[order[b]].Best })
	spared := map[int]bool{}
	for _, i := range order[:min(2, n)] {
		spared[i] = true
	}
	weights := make([]float64, n)
	total := 0.0
	for i, sp := range p.Species {
		if p.Gen-sp.Improved > p.cfg.Stagnation && !spared[i] {
			continue
		}
		for _, g := range sp.Members {
			weights[i] += g.Fitness - lo + 1e-3
		}
		weights[i] /= float64(len(sp.Members))
		total += weights[i]
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		for i := range weights {
			weights[i] = 1
		}
		total = float64(n)
	}
	shares := make([]int, n)
	frac := make([]float64, n)
	given := 0
	for i, w := range weights {
		x := w / total * float64(p.cfg.Pop)
		shares[i] = int(x)
		frac[i] = x - float64(shares[i])
		given += shares[i]
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return frac[idx[a]] > frac[idx[b]] })
	for k := 0; given < p.cfg.Pop; k++ {
		i := idx[k%n]
		if weights[i] > 0 {
			shares[i]++
			given++
		}
	}
	return shares
}

// breed makes the next generation, species by species.
func (p *Population) breed(shares []int) []*neat.Genome {
	children := make([]*neat.Genome, 0, p.cfg.Pop)
	for si, sp := range p.Species {
		share := shares[si]
		if share == 0 {
			continue
		}
		ms := append([]*neat.Genome(nil), sp.Members...)
		sort.SliceStable(ms, func(a, b int) bool { return ms[a].Fitness > ms[b].Fitness })
		if len(ms) >= p.cfg.ChampionMin {
			// The champion is carried over untouched: its Origin lets the
			// evaluator average its next score with this one.
			c := ms[0].Copy()
			c.Fitness = ms[0].Fitness
			c.Origin = ms[0].Lineage()
			c.ID = p.id()
			children = append(children, c)
			share--
		}
		parents := ms[:max(1, int(math.Ceil(p.cfg.Survival*float64(len(ms)))))]
		for ; share > 0; share-- {
			var c *neat.Genome
			if rand.Float64() < p.cfg.MutateOnly {
				c = parents[rand.Intn(len(parents))].Copy()
				c.ID = p.id()
			} else {
				a := parents[rand.Intn(len(parents))]
				b := parents[rand.Intn(len(parents))]
				if rand.Float64() < p.cfg.Interspecies && len(p.Species) > 1 {
					other := p.Species[rand.Intn(len(p.Species))]
					b = other.Members[rand.Intn(len(other.Members))]
				}
				if b.Fitness > a.Fitness {
					a, b = b, a
				}
				c = neat.Crossover(a, b, p.id())
			}
			c.Mutate()
			if len(c.Traits) == 0 {
				c.Traits = neat.RandomTraits(1)
			}
			children = append(children, c)
		}
	}
	return children
}

func (p *Population) id() int {
	p.nextID++
	return p.nextID - 1
}
