package gpubrain

import (
	"math"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"sync"
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

// run does the rest of a round: Start, then Finish.
func run(t *testing.T, b *Batch) {
	t.Helper()
	if err := b.Start(); err != nil {
		t.Fatal(err)
	}
	if err := b.Finish(); err != nil {
		t.Fatal(err)
	}
}

// oversized grows a genome past a slot. Mutate alone takes minutes to get
// there, so every call adds a node, then the rates are put back.
func oversized(seed int64) *neat.Genome {
	g := grown(seed, 2000)
	add, mem, rm := neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate
	defer func() { neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate = add, mem, rm }()
	neat.AddNodeRate, neat.AddMemoryRate, neat.RemoveNodeRate = 1, 0, 0
	for len(g.Nodes) <= MaxNodes {
		g.Mutate()
	}
	return g
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
		b.Open(len(asked))
		ins := make([][]float32, len(asked))
		tickets := make([]int, len(asked))
		for i, e := range asked {
			for k := range in {
				in[k] = float64(float32(r.Float64()*2 - 1))
			}
			tickets[i] = b.Add(e.gpu, in)
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
			got := b.Out(tickets[i])
			for k := range want {
				diff := math.Abs(float64(got[k] - want[k]))
				worst = max(worst, diff)
				if round < 10 && diff > 1e-5 {
					t.Fatalf("round %d output %d: GPU %v, CPU %v", round, k, got[k], want[k])
				}
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
		b.Open(1)
		b.Add(old, in)
		run(t, b)
	}
	for range 3 { // nobody asks: old's slot expires
		b.Open(0)
		run(t, b)
	}
	g := grown(6, 2000)
	young, ref := g.BuildNetwork(), g.BuildNetwork()
	b.Open(1)
	ticket := b.Add(young, in)
	run(t, b)
	in32 := make([]float32, 67)
	for k := range in32 {
		in32[k] = 0.5
	}
	want := ref.Activate(in32)
	for k, x := range b.Out(ticket) {
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
	g := oversized(7)
	big, ref := g.BuildNetwork(), g.BuildNetwork()
	in := make([]float64, 67)
	b.Open(1)
	ticket := b.Add(big, in)
	run(t, b)
	want := ref.Activate(make([]float32, 67))
	for k, x := range b.Out(ticket) {
		if x != want[k] {
			t.Fatalf("oversized output %d: %v, want %v", k, x, want[k])
		}
	}
	if up, _, cpu := b.Stats(); cpu != 1 || up != 0 {
		t.Fatalf("%d networks on CPU, %d uploaded, want 1 and 0", cpu, up)
	}
}

// TestLayoutMatchesShader checks that the capacities the shader is compiled
// with are the ones layout.go packs for.
func TestLayoutMatchesShader(t *testing.T) {
	src, err := os.ReadFile("batch.go")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"MAX_NODES": MaxNodes, "MAX_EDGES": MaxEdges, "MAX_PLASTIC": MaxPlastic, "MAX_VALUES": MaxValues}
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

// TestConcurrentAddMixed calls Add from 8 goroutines with far more births than
// the staging buffer and the arena start with, and with networks too big for a
// slot among the live requests, so the kernel has to skip those.
func TestConcurrentAddMixed(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	const normal, big = 30, 4
	var gpu, cpu []*neat.Network
	for i := range normal {
		g := grown(int64(100+i), 2000)
		gpu, cpu = append(gpu, g.BuildNetwork()), append(cpu, g.BuildNetwork())
	}
	og := oversized(9)
	for range big {
		gpu, cpu = append(gpu, og.BuildNetwork()), append(cpu, og.BuildNetwork())
	}
	// Interleave, so oversized requests sit among the live ones.
	order := rand.New(rand.NewSource(3)).Perm(len(gpu))
	r := rand.New(rand.NewSource(4))
	for round := range 3 {
		ins := make([][]float64, len(order))
		for i := range ins {
			ins[i] = make([]float64, 67)
			for k := range ins[i] {
				ins[i][k] = float64(float32(r.Float64()*2 - 1))
			}
		}
		b.Open(len(order))
		idx := make([]int, len(order))
		var wg sync.WaitGroup
		for w := range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := w; i < len(order); i += 8 {
					idx[i] = b.Add(gpu[order[i]], ins[i])
				}
			}()
		}
		wg.Wait()
		run(t, b)
		bound := 1e-5
		if round > 0 {
			bound = 1e-3
		}
		for i, n := range order {
			in32 := make([]float32, 67)
			for k, x := range ins[i] {
				in32[k] = float32(x)
			}
			want, got := cpu[n].Activate(in32), b.Out(idx[i])
			for k := range want {
				if diff := math.Abs(float64(got[k] - want[k])); diff > bound {
					t.Fatalf("round %d request %d output %d: GPU %v, CPU %v", round, i, k, got[k], want[k])
				}
			}
		}
		up, _, onCPU := b.Stats()
		if round == 0 && (up != normal || onCPU != big) {
			t.Fatalf("uploaded %d, CPU %d: want %d and %d", up, onCPU, normal, big)
		}
		if round > 0 && up != 0 {
			t.Fatalf("round %d uploaded %d networks again", round, up)
		}
	}
}

// TestBackCopiesAreDense checks that the previous-tick values the kernel keeps
// are numbered 1..count without gaps, that a packed network only addresses
// values below nodes+count, and that its kind words still hold the kind.
func TestBackCopiesAreDense(t *testing.T) {
	// rand.Seed is a no-op in recent Go, so grown is not reproducible and a
	// given network may have no back edge: draw until one has.
	var f *neat.Flat
	var at []uint32
	var count int
	for seed := int64(3); count == 0 && seed < 200; seed++ {
		f = grown(seed, 2000).BuildNetwork().Flat()
		at, count = backCopies(f)
	}
	n := f.Nodes()
	seen := make([]bool, count+1)
	for _, a := range at {
		if a != 0 {
			if int(a) > count || seen[a] {
				t.Fatalf("copy %d of %d is out of range or numbered twice", a, count)
			}
			seen[a] = true
		}
	}
	if count == 0 {
		t.Fatal("a grown network with no back edge: the test checks nothing")
	}
	dst := make([]uint32, SlotWords)
	pack(dst, f)
	for i := range f.From {
		if from := dst[edgeOff+2*i]; int(from) >= n+count {
			t.Fatalf("edge %d reads value %d, past the %d+%d the kernel keeps", i, from, n, count)
		}
	}
	for i, k := range f.Kind {
		if dst[kindOff+i]&0xFF != uint32(k) {
			t.Fatalf("node %d: kind word %#x lost kind %d", i, dst[kindOff+i], k)
		}
	}
}

// TestOverflowRunsOnCPUThenGrows asks for more requests than the round has
// room for, from 8 goroutines: every output must match a CPU twin, the
// overflow must show in Stats, and the next Open must have grown so the same
// load fits.
func TestOverflowRunsOnCPUThenGrows(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 3, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	const n = 40
	r := rand.New(rand.NewSource(5))
	for round := range 3 {
		// Fresh networks each round: one that overflows runs on its CPU copy,
		// which does not follow what the card has learned.
		var gpu, cpu []*neat.Network
		for i := range n {
			g := grown(int64(300+round*n+i), 2000)
			gpu, cpu = append(gpu, g.BuildNetwork()), append(cpu, g.BuildNetwork())
		}
		ins := make([][]float64, n)
		for i := range ins {
			ins[i] = make([]float64, 67)
			for k := range ins[i] {
				ins[i][k] = float64(float32(r.Float64()*2 - 1))
			}
		}
		b.Open(4)
		idx := make([]int, n)
		var wg sync.WaitGroup
		for w := range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := w; i < n; i += 8 {
					idx[i] = b.Add(gpu[i], ins[i])
				}
			}()
		}
		wg.Wait()
		run(t, b)
		seen := map[int]bool{}
		for i := range n {
			if k := idx[i] & (1<<genShift - 1); seen[k] || k >= n {
				t.Fatalf("round %d: request %d got index %d twice or out of range", round, i, k)
			} else {
				seen[k] = true
			}
			in32 := make([]float32, 67)
			for k, x := range ins[i] {
				in32[k] = float32(x)
			}
			want, got := cpu[i].Activate(in32), b.Out(idx[i])
			for k := range want {
				if diff := math.Abs(float64(got[k] - want[k])); diff > 1e-3 {
					t.Fatalf("round %d request %d output %d: GPU %v, CPU %v", round, i, k, got[k], want[k])
				}
			}
		}
		_, _, onCPU := b.Stats()
		switch round {
		case 0:
			if onCPU != n-8 {
				t.Fatalf("round 0: %d run on the CPU, want %d", onCPU, n-8)
			}
		case 1:
			if onCPU != 0 {
				t.Fatalf("round 1: %d still overflow after growing", onCPU)
			}
		}
	}
}

// TestOutSurvivesNextRound reads the outputs of round r while round r+1 is
// open and receiving Adds, as the scheduler's workers do. They must match
// the CPU twin (checked once, right after round r) and stay exactly the same
// during round r+1. The rounds overflow and grow their buffers, and one
// network is too big for a slot.
func TestOutSurvivesNextRound(t *testing.T) {
	d := device(t)
	b, err := New(d, 72, 8, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	const n = 20
	og := oversized(11)
	r := rand.New(rand.NewSource(6))
	var prevTickets []int
	var snap [][]float32 // outputs of the previous round, copied right after it
	for round := range 6 {
		// Fresh networks each round: one that overflows runs on its CPU copy,
		// which does not follow what the card has learned.
		var gpu, cpu []*neat.Network
		for i := range n {
			g := grown(int64(500+round*n+i), 2000)
			gpu, cpu = append(gpu, g.BuildNetwork()), append(cpu, g.BuildNetwork())
		}
		gpu, cpu = append(gpu, og.BuildNetwork()), append(cpu, og.BuildNetwork())
		ins := make([][]float64, len(gpu))
		for i := range ins {
			ins[i] = make([]float64, 67)
			for k := range ins[i] {
				ins[i][k] = float64(float32(r.Float64()*2 - 1))
			}
		}
		b.Open(4) // small: rounds overflow, and the buffers grow
		tickets := make([]int, len(gpu))
		var wg sync.WaitGroup
		for w := range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := w; i < len(gpu); i += 4 {
					tickets[i] = b.Add(gpu[i], ins[i])
				}
			}()
		}
		for i, tk := range prevTickets {
			got := b.Out(tk)
			for k, want := range snap[i] {
				if got[k] != want {
					t.Errorf("round %d: output %d of request %d changed once the next round opened: %v, was %v", round, k, i, got[k], want)
				}
			}
		}
		wg.Wait()
		run(t, b)
		snap = snap[:0]
		for i := range gpu {
			in32 := make([]float32, 67)
			for k, x := range ins[i] {
				in32[k] = float32(x)
			}
			want := cpu[i].Activate(in32)
			got := b.Out(tickets[i])
			for k := range want {
				if diff := math.Abs(float64(got[k] - want[k])); diff > 1e-3 {
					t.Fatalf("round %d request %d output %d: GPU %v, CPU %v", round, i, k, got[k], want[k])
				}
			}
			snap = append(snap, append([]float32(nil), got[:8]...))
		}
		prevTickets = tickets
	}
}
