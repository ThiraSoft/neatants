package gpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
	"github.com/ThiraSoft/neatants/neat"
)

func roundHalf(x float32) float32 { return halfToFloat(floatToHalf(x)) }

func TestFloatToHalf(t *testing.T) {
	for _, x := range []float32{0, 1, -1, 0.5, 3.14159, -7.99, 1e-6, 65504, 0.000061} {
		got := roundHalf(x)
		if math.Abs(float64(got-x)) > math.Abs(float64(x))*1e-3+1e-7 {
			t.Fatalf("%g -> %g", x, got)
		}
	}
	if floatToHalf(1) != 0x3C00 || floatToHalf(-2) != 0xC000 {
		t.Fatal("bad encoding")
	}
}

func TestXentMatchesCPU(t *testing.T) {
	t.Run("scalar/D32", func(t *testing.T) { checkXent(t, 32, 0) })
	t.Run("scalar/D128", func(t *testing.T) { checkXent(t, 128, 0) })
	t.Run("coop/D128", func(t *testing.T) {
		if !device(t).Coopmat() {
			t.Skip("no cooperative matrices")
		}
		checkXent(t, 128, xentCoopWave)
	})
}

// At a wave width it was not written for, xent_coop would divide its rows
// wrongly; it must say so with NaN on every row instead of returning scores.
func TestXentCoopRefusesOtherWave(t *testing.T) {
	if !device(t).Coopmat() {
		t.Skip("no cooperative matrices")
	}
	checkXent(t, 128, 32)
}

// checkXent runs xent on random rows and compares with the CPU. wave 0 runs
// the scalar kernel, xentCoopWave the matrix one as Evaluator builds it, and
// any other wave the matrix one at that width, which must refuse.
func checkXent(t *testing.T, D int, wave uint32) {
	dev := device(t)
	const R, V, scored, windows, warm, L = 1000, 300, 50, 5, 16, 66
	genomes := R / (scored * windows)
	d := model.Synthetic(V, D, 2000, 1)
	for i := range d.E {
		d.E[i] = roundHalf(d.E[i])
	}
	starts := []int{0, 300, 700, 1100, 1900 - L}
	scales := []float32{0.2, 0.5, 1, 3}
	flats := make([]*neat.Flat, genomes)
	for i := range flats {
		flats[i] = model.Grown(int64(i), D, 10*i).BuildNetwork().Flat()
	}
	gen := buildGen(flats, scales, starts)

	rng := rand.New(rand.NewSource(7))
	rows := make([]float32, R*D)
	for i := range rows {
		rows[i] = roundHalf(rng.Float32()*2 - 1)
	}
	toks := make([]uint32, len(d.Train))
	for i, x := range d.Train {
		toks[i] = uint32(x)
	}
	bytesOf := func(w []uint32) []byte { return unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w)) }
	up := func(w []uint32) *vk.Buffer {
		b, err := upload(dev, bytesOf(w), vk.UsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Close)
		return b
	}
	genB, tokB, embB, rowsB := up(gen.words), up(toks), up(packHalves(d.E)), up(packHalves(rows))
	size := 4 * (R + 64)
	bitsB, err := dev.Local(size, vk.UsageStorage|vk.UsageTransferSrc|vk.UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer bitsB.Close()
	out, err := dev.Readback(size, vk.UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	coop, tile := wave != 0, 64
	if coop {
		tile = xentCoopRows
	}
	pipe, err := newXentPipe(dev, coop, wave)
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	set, err := pipe.NewSet([]*vk.Buffer{genB, tokB, embB, rowsB, bitsB})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	push := xentPush{R, V, uint32(D), scored, windows, warm, uint32(gen.startsOff), uint32(gen.scaleOff)}
	err = dev.Submit(func(r *vk.Recorder) {
		r.Fill(bitsB, math.Float32bits(-7))
		r.Barrier()
		if coop {
			r.DispatchWide(set, 1, uint32((R+tile-1)/tile), unsafe.Pointer(&push))
		} else {
			r.Dispatch(set, uint32((R+tile-1)/tile), unsafe.Pointer(&push))
		}
		r.Barrier()
		r.Copy(out, 0, bitsB, size)
	})
	if err != nil {
		t.Fatal(err)
	}
	got := unsafe.Slice((*float32)(unsafe.Pointer(&out.Bytes()[0])), R+64)
	if coop && wave != xentCoopWave {
		for r := range R {
			if !math.IsNaN(float64(got[r])) {
				t.Fatalf("wave %d: row %d scored %g instead of NaN", wave, r, got[r])
			}
		}
		return
	}
	worst := 0.0
	for r := range R {
		p := r / scored
		g, w := p/windows, p%windows
		tick := warm + r%scored
		target := d.Train[starts[w]+tick+1]
		want := ref.Bits(d, rows[r*D:(r+1)*D], scales[g], target)
		if math.IsNaN(float64(got[r])) {
			t.Fatalf("row %d is NaN", r)
		}
		worst = math.Max(worst, math.Abs(float64(got[r])-want))
	}
	t.Logf("worst difference %g bits", worst)
	if worst > 1e-3 {
		t.Fatalf("worst difference %g bits", worst)
	}
	for i := R; i < R+64; i++ {
		if got[i] != -7 {
			t.Fatalf("row %d past the end was written: %g", i, got[i])
		}
	}
}
