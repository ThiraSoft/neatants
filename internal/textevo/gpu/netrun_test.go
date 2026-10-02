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

	pipe, err := dev.NewPipeline(netrunSPV, 4, 7*4)
	if err != nil {
		return nil, err
	}
	set, err := pipe.NewSet([]*vk.Buffer{genB, tokB, embB, rowsB})
	if err != nil {
		return nil, err
	}
	push := [7]uint32{uint32(pairs), uint32(len(starts)), uint32(length), uint32(warm), uint32(d.Dim),
		uint32(gen.startsOff), uint32(gen.goffOff)}
	err = dev.Submit(func(r *vk.Recorder) {
		r.Dispatch(set, uint32(pairs), unsafe.Pointer(&push))
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
