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
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/gpu"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// esWeightMax is the bound neat keeps link weights in.
const esWeightMax = 8

// runES keeps the topology of a softmax champion and tunes the weights of
// its enabled links with OpenAI-ES: each step scores -pop/2 antithetic pairs
// of Gaussian perturbations on the same windows, and moves the weights along
// the perturbations weighted by their centered ranks, through Adam. It tells
// whether the weights or the structure hold the runs back.
func runES() {
	d := load()
	raw, err := os.ReadFile(*esOf)
	if err != nil {
		fail(1, "%v", err)
	}
	var c champion
	if err := json.Unmarshal(raw, &c); err != nil {
		fail(1, "%s: %v", *esOf, err)
	}
	if c.Genome == nil || c.Score != "" {
		fail(1, "%s: -es needs a softmax champion", *esOf)
	}
	if c.Dim != d.Dim || c.Checksum != prep.Checksum(d) {
		fail(1, "%s was evolved on another corpus preparation", *esOf)
	}
	if *pop < 2 || *pop%2 != 0 {
		fail(2, "-pop must be even under -es: perturbations come in pairs")
	}
	if *warm >= *length {
		fail(2, "-warm must be smaller than -len")
	}
	model.ShapeOf(c.Genome, d.Dim)

	var ev evaluator = cpuEval{d, nil}
	if !*cpu {
		dev, err := vk.Open()
		if err != nil {
			fail(1, "no Vulkan device (%v), try -cpu", err)
		}
		defer dev.Close()
		ge, err := gpu.New(dev, d)
		if err != nil {
			fail(1, "%v", err)
		}
		defer ge.Close()
		ev = ge
	}

	dir := *out
	if dir == "" {
		dir = filepath.Join("runs", time.Now().Format("2006-01-02_15-04-05")+"-es")
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
	head := []string{"step", "center_bpb", "mean_bpb", "best_bpb", "eval_ms", "total_ms", "val_bpb"}
	w.Write(head)
	fmt.Printf("es on %s (val %.4f, gen %d), sigma %g, lr %g, pairs %d\n", *esOf, c.ValBPB, c.Gen, *esSigma, *esLR, *pop/2)
	fmt.Println(strings.Join(head, ","))

	var stop atomic.Bool
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	go func() { <-sig; stop.Store(true) }()

	center := c.Genome.Copy()
	var on []int
	for i, cg := range center.Conns {
		if cg.Enabled {
			on = append(on, i)
		}
	}
	n := len(on)
	theta := make([]float64, n)
	for j, i := range on {
		theta[j] = center.Conns[i].Weight
	}
	// Adam state.
	m, v := make([]float64, n), make([]float64, n)
	const b1, b2 = 0.9, 0.999

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	pairs := *pop / 2
	eps := make([][]float64, pairs)
	for k := range eps {
		eps[k] = make([]float64, n)
	}
	// The last genome is the unperturbed center, scored on the same windows.
	gs := make([]*neat.Genome, *pop+1)
	for k := range gs {
		gs[k] = center.Copy()
		gs[k].ID = k + 1
	}
	set := func(g *neat.Genome, sign float64, e []float64) {
		for j, i := range on {
			x := theta[j]
			if e != nil {
				x += sign * *esSigma * e[j]
			}
			g.Conns[i].Weight = math.Max(-esWeightMax, math.Min(esWeightMax, x))
		}
	}
	bestVal := c.ValBPB
	grad := make([]float64, n)
	for step := 1; ; step++ {
		t0 := time.Now()
		for k := range pairs {
			for j := range eps[k] {
				eps[k][j] = rng.NormFloat64()
			}
			set(gs[2*k], 1, eps[k])
			set(gs[2*k+1], -1, eps[k])
		}
		set(gs[*pop], 0, nil)
		starts := model.DrawStarts(rng, len(d.Train), *windows, *length)
		bpb, _, err := ev.Evaluate(gs, starts, *length, *warm)
		if err != nil {
			fail(1, "step %d: %v", step, err)
		}
		evalTime := time.Since(t0)

		// Centered ranks: the best perturbation weighs 0.5, the worst -0.5,
		// whatever the scale of the scores.
		idx := make([]int, *pop)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return bpb[idx[a]] > bpb[idx[b]] })
		rank := make([]float64, *pop)
		for r, i := range idx {
			rank[i] = float64(r)/float64(*pop-1) - 0.5
		}
		clear(grad)
		for k := range pairs {
			s := rank[2*k] - rank[2*k+1]
			for j, e := range eps[k] {
				grad[j] += s * e
			}
		}
		lr := *esLR * math.Sqrt(1-math.Pow(b2, float64(step))) / (1 - math.Pow(b1, float64(step)))
		for j := range theta {
			g := grad[j] / (float64(*pop) * *esSigma)
			m[j] = b1*m[j] + (1-b1)*g
			v[j] = b2*v[j] + (1-b2)*g*g
			theta[j] = math.Max(-esWeightMax, math.Min(esWeightMax, theta[j]+lr*m[j]/(math.Sqrt(v[j])+1e-8)))
		}

		mean, best := 0.0, math.Inf(1)
		for _, b := range bpb[:*pop] {
			mean += b
			best = math.Min(best, b)
		}
		mean /= float64(*pop)
		last := *gens > 0 && step >= *gens
		halt := stop.Load()
		val := ""
		if step%10 == 0 || last || halt {
			set(center, 0, nil)
			x, err := ev.Validate(center)
			if err != nil {
				fail(1, "validation: %v", err)
			}
			val = num(x)
			if x < bestVal {
				bestVal = x
				if err := save(dir, champion{center, d.Dim, c.Banks, "", "", c.Checksum, x, c.Gen}); err != nil {
					fail(1, "%v", err)
				}
			}
		}
		row := []string{strconv.Itoa(step), num(bpb[*pop]), num(mean), num(best),
			strconv.FormatInt(evalTime.Milliseconds(), 10), strconv.FormatInt(time.Since(t0).Milliseconds(), 10), val}
		w.Write(row)
		w.Flush()
		fmt.Println(strings.Join(row, ","))
		if last || halt {
			return
		}
	}
}
