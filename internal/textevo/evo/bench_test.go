package evo

import (
	"encoding/json"
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/neatants/neat"
)

// BenchmarkNextRealistic times a generation of evo at the size real runs
// reach: 200 mutated copies of the first p200 champion (2389 edges).
func BenchmarkNextRealistic(b *testing.B) {
	raw, err := os.ReadFile("../../../runs/first-p200/champion.json")
	if err != nil {
		b.Skip(err)
	}
	var c struct {
		Genome *neat.Genome `json:"genome"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		b.Fatal(err)
	}
	cfg := DefaultConfig()
	cfg.Pop = 200
	p := New(cfg, 128)
	for i := range p.Genomes {
		g := c.Genome.Copy()
		g.ID = i + 1
		g.Mutate()
		p.Genomes[i] = g
	}
	b.ResetTimer()
	for range b.N {
		f := make([]float64, len(p.Genomes))
		for i := range f {
			f[i] = rand.Float64()
		}
		p.Next(f)
	}
}

// BenchmarkNextDense times a generation of a dense first population at D 128
// (every output reads every input, 16512 connections a genome).
func BenchmarkNextDense(b *testing.B) {
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	neat.MinimalLinks = 128
	cfg := DefaultConfig()
	cfg.Pop = 200
	p := New(cfg, 128)
	b.ResetTimer()
	for range b.N {
		f := make([]float64, len(p.Genomes))
		for i := range f {
			f[i] = rand.Float64()
		}
		p.Next(f)
	}
}
