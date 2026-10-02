package sim

import (
	"github.com/ThiraSoft/neatants/neat"
	"math"
	"math/rand"
)

// NEAT speciation, adapted to a steady-state world as in rtNEAT: species keep
// a fixed representative, and at every birth the breeding pool is sorted into
// them. A species is picked with a probability proportional to the mean
// fitness of its members, which is the sum of their shared fitness (raw
// fitness divided by the species size): a big species gets no more children
// than a small one of the same quality, so new structures have time to tune
// their weights before competing with the established ones.

const (
	// A species whose best fitness has not risen for this long no longer
	// breeds (NEAT: 15 generations; an ant lives about 15k ticks).
	stagnationTicks = 15 * 15000
	// Species at least this big have their champion copied unchanged.
	championMinSize = 5
	// Share of crossovers made with a mate of another species.
	interspeciesRate = 0.001
	// The compatibility threshold drifts to keep this many species: with a
	// fixed one, a pool ends up as one big species (no protection) or as
	// lone genomes (no crossover), depending on how its genomes spread.
	speciesMin, speciesMax = 3, 8
	distStep               = 0.03
)

type Species struct {
	ID       int            `json:"id"`
	Rep      *neat.Genome   `json:"rep"`
	Best     float64        `json:"best"`
	Improved int            `json:"improved"` // evolution tick when Best last rose
	Members  []*neat.Genome `json:"-"`
}

// Speciation is a pool's species registry. It is saved with the pool so
// that stagnation and the threshold survive a reload.
type Speciation struct {
	List   []*Species `json:"list"`
	NextID int        `json:"next_id"`
	Dist   float64    `json:"dist"`
}

// Threshold is the current compatibility threshold of the pool.
func (s *Speciation) Threshold() float64 {
	if s == nil || s.Dist <= 0 {
		return speciesDist
	}
	return s.Dist
}

// assign sorts the pool into species, creates species for the genomes that
// fit none, drops the empty ones and records the progress of each.
func (s *Speciation) assign(pool []*neat.Genome, now int) {
	for _, sp := range s.List {
		sp.Members = sp.Members[:0]
	}
	dist := s.Threshold()
	for _, g := range pool {
		var home *Species
		for _, sp := range s.List {
			if neat.Compatibility(g, sp.Rep) < dist {
				home = sp
				break
			}
		}
		if home == nil {
			s.NextID++
			home = &Species{ID: s.NextID, Rep: g, Best: g.Fitness, Improved: now}
			s.List = append(s.List, home)
		}
		home.Members = append(home.Members, g)
	}
	alive := s.List[:0]
	for _, sp := range s.List {
		if len(sp.Members) == 0 {
			continue
		}
		for _, g := range sp.Members {
			if g.Fitness > sp.Best {
				sp.Best, sp.Improved = g.Fitness, now
			}
		}
		alive = append(alive, sp)
	}
	s.List = alive
	switch {
	case len(s.List) < speciesMin:
		s.Dist = max(0.3, dist*(1-distStep))
	case len(s.List) > speciesMax:
		s.Dist = min(6, dist*(1+distStep))
	default:
		s.Dist = dist
	}
}

// pick draws the species that raises the next child. Stagnant species are
// skipped, except the best one so that breeding never stops.
func (s *Speciation) pick(pool []*neat.Genome, now int) *Species {
	s.assign(pool, now)
	var best *Species
	for _, sp := range s.List {
		if best == nil || sp.Best > best.Best {
			best = sp
		}
	}
	var cands []*Species
	var weights []float64
	total := 0.0
	for _, sp := range s.List {
		if sp != best && now-sp.Improved > stagnationTicks {
			continue
		}
		mean := 0.0
		for _, g := range sp.Members {
			mean += math.Max(0, g.Fitness)
		}
		mean = mean/float64(len(sp.Members)) + 0.01
		cands = append(cands, sp)
		weights = append(weights, mean)
		total += mean
	}
	r := rand.Float64() * total
	for i, sp := range cands {
		if r -= weights[i]; r <= 0 {
			return sp
		}
	}
	return cands[len(cands)-1]
}

func (sp *Species) champion() *neat.Genome {
	best := sp.Members[0]
	for _, g := range sp.Members[1:] {
		if g.Fitness > best.Fitness {
			best = g
		}
	}
	return best
}

// breedSpeciated makes a child from one species of the pool: its champion
// copied unchanged now and then, otherwise a mutated copy or a crossover of
// two members (rarely with another species).
func breedSpeciated(s *Speciation, pool []*neat.Genome, now int) *neat.Genome {
	sp := s.pick(pool, now)
	if rand.Float64() < eliteRate {
		if len(sp.Members) >= championMinSize {
			champ := sp.champion()
			child := champ.Copy()
			child.Origin = champ.Lineage() // evaluated again, its fitness averaged
			return child
		}
		child := sp.champion().Copy()
		child.Mutate()
		return child
	}
	a := tournament(sp.Members, tournamentK)
	mates := make([]*neat.Genome, 0, len(sp.Members))
	src := sp.Members
	if rand.Float64() < interspeciesRate {
		src = pool
	}
	for _, g := range src {
		if g != a {
			mates = append(mates, g)
		}
	}
	if len(mates) == 0 || rand.Float64() >= crossoverRate {
		child := a.Copy()
		child.Mutate()
		return child
	}
	b := tournament(mates, 3)
	if b.Fitness > a.Fitness {
		a, b = b, a
	}
	child := neat.Crossover(a, b, 0)
	child.Mutate()
	return child
}
