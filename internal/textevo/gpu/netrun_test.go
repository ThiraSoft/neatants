package gpu

import (
	"math"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
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

// runNetrun runs netrun alone and returns [genome][window][tick][dim].
func runNetrun(dev *vk.Device, d *prep.Data, gs []*neat.Genome, starts []int, length, warm int) ([][][][]float32, error) {
	flats := make([]*neat.Flat, len(gs))
	scales := make([]float32, len(gs))
	for i, g := range gs {
		flats[i] = g.BuildNetwork().Flat()
		scales[i] = model.LogitScale(g, d.Dim)
	}
	gen := buildGen(flats, scales, starts)
	toks := make([]uint32, len(d.Train))
	for i, x := range d.Train {
		toks[i] = uint32(x)
	}
	bytesOf := func(p unsafe.Pointer, n int) []byte { return unsafe.Slice((*byte)(p), n) }
	genB, err := upload(dev, bytesOf(unsafe.Pointer(&gen.words[0]), 4*len(gen.words)), vk.UsageStorage)
	if err != nil {
		return nil, err
	}
	defer genB.Close()
	tokB, err := upload(dev, bytesOf(unsafe.Pointer(&toks[0]), 4*len(toks)), vk.UsageStorage)
	if err != nil {
		return nil, err
	}
	defer tokB.Close()
	embB, err := upload(dev, bytesOf(unsafe.Pointer(&d.E[0]), 4*len(d.E)), vk.UsageStorage)
	if err != nil {
		return nil, err
	}
	defer embB.Close()

	pairs := len(gs) * len(starts)
	scored := length - warm
	size := pairs * scored * d.Dim * 2
	rowsB, err := dev.Local(size, vk.UsageStorage|vk.UsageTransferSrc)
	if err != nil {
		return nil, err
	}
	defer rowsB.Close()
	out, err := dev.Readback(size, vk.UsageTransferDst)
	if err != nil {
		return nil, err
	}
	defer out.Close()

	pipes, err := newNetPipes(dev)
	if err != nil {
		return nil, err
	}
	var sets [2]*vk.Set
	for c, pipe := range pipes {
		defer pipe.Close()
		if sets[c], err = pipe.NewSet([]*vk.Buffer{genB, tokB, embB, rowsB}); err != nil {
			return nil, err
		}
		defer sets[c].Close()
	}
	err = dev.Submit(func(r *vk.Recorder) {
		recordNet(r, sets, gen, len(starts), length, warm, d.Dim)
		r.Barrier()
		r.Copy(out, 0, rowsB, size)
	})
	if err != nil {
		return nil, err
	}
	raw := out.Bytes()
	h := func(i int) float32 { return halfToFloat(uint16(raw[2*i]) | uint16(raw[2*i+1])<<8) }
	res := make([][][][]float32, len(gs))
	for gi := range gs {
		res[gi] = make([][][]float32, len(starts))
		for wi := range starts {
			p := gi*len(starts) + wi
			res[gi][wi] = make([][]float32, scored)
			for ti := range scored {
				row := make([]float32, d.Dim)
				for j := range row {
					row[j] = h((p*scored+ti)*d.Dim + j)
				}
				res[gi][wi][ti] = row
			}
		}
	}
	return res, nil
}

func TestNetrunMatchesCPU(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 2000, 1)
	var gs []*neat.Genome
	for i := range 20 {
		gs = append(gs, model.Grown(int64(i), 32, 50*i))
	}
	starts := []int{0, 500, 1300}
	const L, W = 64, 16
	rows, err := runNetrun(dev, d, gs, starts, L, W)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for gi, g := range gs {
		for wi, s := range starts {
			want := ref.Rows(g, d, d.Train, s, L, W)
			for ti := range want {
				for j := range want[ti] {
					worst = math.Max(worst, math.Abs(float64(rows[gi][wi][ti][j]-want[ti][j])))
				}
			}
		}
	}
	t.Logf("worst difference %g", worst)
	// fp16 rows: a half has 11 bits of mantissa, about 5e-4 at 1.
	if worst > 2e-3 {
		t.Fatalf("worst difference %g", worst)
	}
}

// Levels longer than LevelEdges are split by Pack: the run must stay the
// network's.
func TestNetrunSplitLevels(t *testing.T) {
	dev := device(t)
	const D = 300
	d := model.Synthetic(300, D, 2000, 3)
	var gs []*neat.Genome
	for i := range 4 {
		g := model.Grown(int64(i), D, 100*i)
		if !Fits(g.BuildNetwork().Flat()) {
			t.Fatalf("genome %d does not fit", i)
		}
		gs = append(gs, g)
	}
	starts := []int{0, 900}
	const L, W = 48, 16
	rows, err := runNetrun(dev, d, gs, starts, L, W)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for gi, g := range gs {
		for wi, s := range starts {
			want := ref.Rows(g, d, d.Train, s, L, W)
			for ti := range want {
				for j := range want[ti] {
					worst = math.Max(worst, math.Abs(float64(rows[gi][wi][ti][j]-want[ti][j])))
				}
			}
		}
	}
	t.Logf("worst difference %g", worst)
	if worst > 2e-3 {
		t.Fatalf("worst difference %g", worst)
	}
}

// Networks past the small variant run in the big one, next to small ones in
// the same generation, and both give the network's rows.
func TestNetrunBigVariant(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 2000, 4)
	var gs []*neat.Genome
	for i := range 6 {
		g := model.Grown(int64(i), 32, 100)
		if i%2 == 1 {
			for range smallMemory + 20 {
				g.AddMemory()
			}
		}
		if c, want := netClass(g.BuildNetwork().Flat()), i%2; c != want {
			t.Fatalf("genome %d runs in variant %d, want %d", i, c, want)
		}
		gs = append(gs, g)
	}
	starts := []int{0, 500, 1300}
	const L, W = 48, 16
	rows, err := runNetrun(dev, d, gs, starts, L, W)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for gi, g := range gs {
		for wi, s := range starts {
			want := ref.Rows(g, d, d.Train, s, L, W)
			for ti := range want {
				for j := range want[ti] {
					worst = math.Max(worst, math.Abs(float64(rows[gi][wi][ti][j]-want[ti][j])))
				}
			}
		}
	}
	t.Logf("worst difference %g", worst)
	if worst > 2e-3 {
		t.Fatalf("worst difference %g", worst)
	}
}

// A dense first generation (every output reads all D inputs and the bias)
// at D = 128 puts 128 outputs of 129 edges in one level, 16512 edges: Pack
// must split it so that each piece fits LevelEdges, and the run must stay
// the network's.
func TestNetrunDenseStart(t *testing.T) {
	dev := device(t)
	const D = 128
	defer func(k int) { neat.MinimalLinks = k }(neat.MinimalLinks)
	neat.MinimalLinks = D
	d := model.Synthetic(300, D, 2000, 5)
	var gs []*neat.Genome
	for i := range 4 {
		g := model.NewGenome(i+1, D)
		for range 20 * i {
			g.Mutate()
		}
		f := g.BuildNetwork().Flat()
		if c := netClass(f); c != 0 {
			t.Fatalf("genome %d runs in variant %d, want the small one", i, c)
		}
		if i == 0 {
			if len(f.From) != D*(D+1) {
				t.Fatalf("%d edges, want %d", len(f.From), D*(D+1))
			}
			if u := checkRecord(t, f, Pack(nil, f)); u.nLevels < D*(D+1)/LevelEdges {
				t.Fatalf("the output level was split in %d, want at least %d", u.nLevels, D*(D+1)/LevelEdges)
			}
		}
		gs = append(gs, g)
	}
	starts := []int{0, 900}
	const L, W = 48, 16
	rows, err := runNetrun(dev, d, gs, starts, L, W)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for gi, g := range gs {
		for wi, s := range starts {
			want := ref.Rows(g, d, d.Train, s, L, W)
			for ti := range want {
				for j := range want[ti] {
					worst = math.Max(worst, math.Abs(float64(rows[gi][wi][ti][j]-want[ti][j])))
				}
			}
		}
	}
	t.Logf("worst difference %g", worst)
	if worst > 2e-3 {
		t.Fatalf("worst difference %g", worst)
	}
}

// Networks with state banks: the kernel keeps the banks and the gates across
// the window, and the rows are the D predictions only.
func TestNetrunStateMatchesCPU(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(300, 32, 2000, 21)
	var gs []*neat.Genome
	for i := range 20 {
		s := model.Shape{Dim: 32, Banks: 2}
		if i%4 == 3 {
			s.Banks = 4
		}
		gs = append(gs, model.GrownShape(int64(i), s, 50*i))
	}
	starts := []int{0, 500, 1300}
	const L, W = 64, 16
	rows, err := runNetrun(dev, d, gs, starts, L, W)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for gi, g := range gs {
		for wi, s := range starts {
			want := ref.Rows(g, d, d.Train, s, L, W)
			for ti := range want {
				for j := range want[ti] {
					worst = math.Max(worst, math.Abs(float64(rows[gi][wi][ti][j]-want[ti][j])))
				}
			}
		}
	}
	t.Logf("worst difference %g", worst)
	if worst > 2e-3 {
		t.Fatalf("worst difference %g", worst)
	}
}
