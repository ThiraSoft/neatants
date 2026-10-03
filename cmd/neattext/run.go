package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/evo"
	"github.com/ThiraSoft/neatants/internal/textevo/gpu"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
	"github.com/ThiraSoft/neatants/neat"
)

// evaluator scores genomes in bits per byte, on the GPU or on the CPU.
type evaluator interface {
	Evaluate(gs []*neat.Genome, starts []int, length, warm int) ([]float64, int, error)
	Validate(g *neat.Genome) (float64, error)
}

// cpuEval is the reference evaluation behind the evaluator interface, with
// the tree scoring when t is set. It has no size limit, so nothing is ever
// oversized.
type cpuEval struct {
	d *prep.Data
	t *model.Tree
}

func (c cpuEval) Evaluate(gs []*neat.Genome, starts []int, length, warm int) ([]float64, int, error) {
	return c.eval(gs, c.d.Train, starts, length, warm), 0, nil
}

func (c cpuEval) Validate(g *neat.Genome) (float64, error) {
	ids := c.d.Val
	return c.eval([]*neat.Genome{g}, ids, model.ValStarts(len(ids)), model.ValLen, model.Warm)[0], nil
}

func (c cpuEval) eval(gs []*neat.Genome, ids []int32, starts []int, length, warm int) []float64 {
	if c.t != nil {
		return ref.EvaluateTree(gs, c.d, c.t, ids, starts, length, warm)
	}
	return ref.Evaluate(gs, c.d, ids, starts, length, warm)
}

var header = []string{"gen", "best_bpb", "mean_bpb", "species", "nodes", "edges", "memory", "plastic", "oversized", "eval_ms", "total_ms", "val_bpb"}

func runEvolve() {
	if *warm >= *length {
		fail(2, "-warm must be smaller than -len")
	}
	if *pop < 1 {
		fail(2, "-pop must be at least 1")
	}
	if *windows < 1 {
		fail(2, "-windows must be at least 1")
	}
	if *pop**windows > 65535 {
		fail(2, "-pop times -windows must not exceed 65535, the Vulkan dispatch limit")
	}
	if *links < 1 {
		fail(2, "-links must be at least 1")
	}
	if *wmut <= 0 {
		fail(2, "-wmut must be positive")
	}
	if *banks < 0 || *banks > model.MaxBanks {
		fail(2, "-state must be between 0 and %d", model.MaxBanks)
	}
	if *score != "softmax" && *score != "tree" {
		fail(2, "-score must be softmax or tree")
	}
	if *input != model.InputEmb && *input != model.InputCode && *input != model.InputBoth {
		fail(2, "-input must be emb, code or both")
	}
	if *input != model.InputEmb && *score != "tree" {
		fail(2, "-input %s needs -score tree", *input)
	}
	// The dimension is the prepared file's: -dim only tells -prep what to
	// write, so a run never has to repeat it.
	d := load()
	if *length+1 >= len(d.Train) {
		fail(2, "-len %d leaves no room in %d train tokens", *length, len(d.Train))
	}

	// rows is what the networks read of a token, d itself unless -input
	// says otherwise; the corpus and the scoring are d's.
	rows := d
	var tr *model.Tree
	saved, savedIn := "", ""
	if *score == "tree" {
		tr = model.BuildTree(d)
		rows = model.InputData(d, tr, *input)
		if tr.Depth > rows.Dim {
			fail(2, "a tree of depth %d needs an embedding at least as wide, -input code reads one", tr.Depth)
		}
		saved = "tree"
		if *input != model.InputEmb {
			savedIn = *input
		}
	}
	shape := model.Shape{Dim: rows.Dim, Banks: *banks}
	if tr != nil && tr.Depth != rows.Dim {
		shape.Code = tr.Depth
	}

	var ev evaluator = cpuEval{rows, tr}
	if !*cpu {
		dev, err := vk.Open()
		if err != nil {
			fail(1, "no Vulkan device (%v), try -cpu", err)
		}
		defer dev.Close()
		var ge *gpu.Evaluator
		if tr != nil {
			ge, err = gpu.NewTree(dev, rows, tr)
		} else {
			ge, err = gpu.New(dev, d)
		}
		if err != nil {
			fail(1, "%v", err)
		}
		defer ge.Close()
		ev = ge
	}

	dir := *out
	if dir == "" {
		dir = filepath.Join("runs", time.Now().Format("2006-01-02_15-04-05"))
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fail(1, "%v", err)
	}
	f, err := os.Create(filepath.Join(dir, "run.csv"))
	if err != nil {
		fail(1, "%v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write(header)
	fmt.Println(strings.Join(header, ","))

	fmt.Printf("data %s, dim %d, links %d, wmut %g, state %d, pop %d, score %s\n", *dataPath, d.Dim, *links, *wmut, *banks, *pop, *score)
	if tr != nil {
		fmt.Printf("tree of depth %d over %d tokens, input %s (%d values a token)\n", tr.Depth, d.Vocab(), *input, rows.Dim)
	}
	printBaselines(d)

	// Ctrl+C lets the generation in flight finish, then validates and saves.
	var stop atomic.Bool
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	go func() { <-sig; stop.Store(true) }()

	neat.HebbRate = *hebb
	neat.MinimalLinks = *links
	neat.WeightsPerMutation = *wmut
	cfg := evo.DefaultConfig()
	cfg.Pop = *pop
	// -links counts the bank inputs like the embedding ones: a fresh output
	// reads that many of all Dim*(1+banks) inputs, drawn at random.
	p := evo.NewShape(cfg, shape)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sum := prep.Checksum(d)
	bestVal := math.Inf(1)

	for gen := 1; ; gen++ {
		t0 := time.Now()
		starts := model.DrawStarts(rng, len(d.Train), *windows, *length)
		bpb, over, err := ev.Evaluate(p.Genomes, starts, *length, *warm)
		if err != nil {
			fail(1, "generation %d: %v", gen, err)
		}
		evalTime := time.Since(t0)
		fit := make([]float64, len(bpb))
		best, mean, finite := math.Inf(1), 0.0, 0
		for i, b := range bpb {
			fit[i] = -b
			if !math.IsInf(b, 0) && !math.IsNaN(b) {
				best = math.Min(best, b)
				mean += b
				finite++
			}
		}
		if finite > 0 {
			mean /= float64(finite)
		} else {
			mean = math.NaN()
		}
		p.Next(fit)
		champ := p.Best()
		flat := champ.BuildNetwork().Flat()
		memory := 0
		for _, k := range flat.Kind {
			if k == uint8(neat.Memory) {
				memory++
			}
		}

		last := *gens > 0 && gen >= *gens
		halt := stop.Load()
		val := ""
		if gen%10 == 0 || last || halt {
			v, err := ev.Validate(champ)
			if err != nil {
				fail(1, "validation: %v", err)
			}
			val = num(v)
			if v < bestVal {
				bestVal = v
				if err := save(dir, champion{champ, d.Dim, *banks, saved, savedIn, sum, v, gen}); err != nil {
					fail(1, "%v", err)
				}
			}
		}
		row := []string{strconv.Itoa(gen), num(best), num(mean), strconv.Itoa(len(p.Species)),
			strconv.Itoa(flat.Nodes()), strconv.Itoa(len(flat.From)), strconv.Itoa(memory), strconv.Itoa(len(flat.Plastic)),
			strconv.Itoa(over), strconv.FormatInt(evalTime.Milliseconds(), 10), strconv.FormatInt(time.Since(t0).Milliseconds(), 10), val}
		w.Write(row)
		w.Flush()
		fmt.Println(strings.Join(row, ","))
		if last || halt {
			return
		}
	}
}

func num(x float64) string { return strconv.FormatFloat(x, 'f', 4, 64) }

// save writes the champion through a temporary file so an interrupted write
// never leaves a truncated champion.json.
func save(dir string, c champion) error {
	raw, err := json.Marshal(c)
	if err != nil {
		return err
	}
	tmp := filepath.Join(dir, "champion.json.tmp")
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, "champion.json"))
}
