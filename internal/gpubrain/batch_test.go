package gpubrain

import (
	"math"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"testing"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

func device(t *testing.T) *vk.Device {
	t.Helper()
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	t.Cleanup(d.Close)
	return d
}

func grown(seed int64, mutations int) *neat.Genome {
	rand.Seed(seed)
	g := neat.NewGenomeWithHidden(1, 67, 8, 8)
	for range mutations {
		g.Mutate()
	}
	return g
}

// TestBatchMatchesCPU runs a population on the GPU and on the CPU for a few
// hundred rounds, with births, deaths (a network stops being requested) and
// think_every 2, and compares every output.
func TestBatchMatchesCPU(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 3, 16) // small capacity: forces the arena to grow
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	type ent struct {
		gpu, cpu *neat.Network
		born     int
	}
	var pop []*ent
	r := rand.New(rand.NewSource(1))
	in := make([]float64, 67)
	worst := 0.0
	for round := range 400 {
		if round%10 == 0 || len(pop) < 40 {
			g := grown(int64(round), 2000)
			pop = append(pop, &ent{g.BuildNetwork(), g.BuildNetwork(), round})
		}
		if round%25 == 24 { // the oldest dies
			pop = pop[1:]
		}
		var asked []*ent
		for i, e := range pop {
			if (round+i)%2 == 0 { // think_every 2
				asked = append(asked, e)
			}
		}
		b.Begin(len(asked))
		ins := make([][]float32, len(asked))
		for i, e := range asked {
			for k := range in {
				in[k] = float64(float32(r.Float64()*2 - 1))
			}
			b.Set(i, e.gpu, in)
			ins[i] = make([]float32, 67)
			for k, x := range in {
				ins[i][k] = float32(x)
			}
		}
		if err := b.Start(); err != nil {
			t.Fatal(err)
		}
		if err := b.Finish(); err != nil {
			t.Fatal(err)
		}
		for i, e := range asked {
			want := e.cpu.Activate(ins[i])
			got := b.Out(i)
			for k := range want {
				worst = max(worst, math.Abs(float64(got[k]-want[k])))
			}
		}
	}
	if worst > 1e-3 {
		t.Fatalf("GPU drifts from CPU by %v", worst)
	}
	t.Logf("largest difference over 400 rounds: %g", worst)
}

// TestSlotReuseStartsFresh frees a slot (its network stops thinking) and
// gives it to a newborn: the newborn must answer like a fresh CPU network.
func TestSlotReuseStartsFresh(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	in := make([]float64, 67)
	for k := range in {
		in[k] = 0.5
	}
	old := grown(5, 2000).BuildNetwork()
	for range 20 {
		b.Begin(1)
		b.Set(0, old, in)
		b.Start()
		b.Finish()
	}
	for range 3 { // nobody asks: old's slot expires
		b.Begin(0)
		b.Start()
		b.Finish()
	}
	g := grown(6, 2000)
	young, ref := g.BuildNetwork(), g.BuildNetwork()
	b.Begin(1)
	b.Set(0, young, in)
	b.Start()
	b.Finish()
	in32 := make([]float32, 67)
	for k := range in32 {
		in32[k] = 0.5
	}
	want := ref.Activate(in32)
	for k, x := range b.Out(0) {
		if math.Abs(float64(x-want[k])) > 1e-5 {
			t.Fatalf("output %d of a reused slot: %v, fresh CPU %v", k, x, want[k])
		}
	}
	if up, freed, _ := b.Stats(); up != 2 || freed != 1 {
		t.Fatalf("uploaded %d, freed %d: want 2 and 1", up, freed)
	}
}

// TestOversizedRunsOnCPU gives the batch a network bigger than a slot.
func TestOversizedRunsOnCPU(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	g := grown(7, 2000)
	// Mutate alone takes minutes to pass the slot, so make every call add a
	// node, then put the rates back.
	add, mem, rm := neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate
	defer func() { neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate = add, mem, rm }()
	neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate = 1, 0, 0
	for len(g.Nodes) <= MaxNodes { // grow past the slot
		g.Mutate()
	}
	big, ref := g.BuildNetwork(), g.BuildNetwork()
	in := make([]float64, 67)
	b.Begin(1)
	b.Set(0, big, in)
	b.Start()
	b.Finish()
	want := ref.Activate(make([]float32, 67))
	for k, x := range b.Out(0) {
		if x != want[k] {
			t.Fatalf("oversized output %d: %v, want %v", k, x, want[k])
		}
	}
	if _, _, cpu := b.Stats(); cpu != 1 {
		t.Fatalf("%d networks on CPU, want 1", cpu)
	}
}

// TestLayoutMatchesShader checks that the capacities the shader is compiled
// with are the ones layout.go packs for.
func TestLayoutMatchesShader(t *testing.T) {
	src, err := os.ReadFile("batch.go")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"MAX_NODES": MaxNodes, "MAX_EDGES": MaxEdges, "MAX_PLASTIC": MaxPlastic}
	for name, v := range want {
		m := regexp.MustCompile("-D" + name + `=(\d+)u`).FindSubmatch(src)
		if m == nil {
			t.Fatalf("no -D%s in the go:generate line", name)
		}
		if n, _ := strconv.Atoi(string(m[1])); n != v {
			t.Fatalf("%s: shader built with %d, layout.go says %d", name, n, v)
		}
	}
	if SlotWords != 18442 {
		t.Fatalf("SlotWords = %d, want 18442", SlotWords)
	}
}
