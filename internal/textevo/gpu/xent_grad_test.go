package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/ref"
	"github.com/ThiraSoft/neatants/neat"
)

// gradRef is the float64 reference of the fused kernel for one row: the bits,
// the gradient with respect to o and the one with respect to the scale.
func gradRef(emb func(j int) []float32, prior []float32, vocab int, o []float32, s float64, target int) (bits float64, dO []float64, dS float64) {
	x := make([]float64, vocab)
	dot := make([]float64, vocab)
	m := math.Inf(-1)
	for j := range x {
		for k, e := range emb(j) {
			dot[j] += float64(e) * float64(o[k])
		}
		x[j] = float64(prior[j]) + s*dot[j]
		m = math.Max(m, x[j])
	}
	sum := 0.0
	for _, v := range x {
		sum += math.Exp(v - m)
	}
	dO = make([]float64, len(o))
	for j := range x {
		p := math.Exp(x[j]-m) / sum
		dS += p * dot[j]
		for k, e := range emb(j) {
			dO[k] += p * float64(e)
		}
	}
	for k, e := range emb(target) {
		dO[k] = s * (dO[k] - float64(e)) / math.Ln2
	}
	dS = (dS - dot[target]) / math.Ln2
	return (m + math.Log(sum) - x[target]) / math.Ln2, dO, dS
}

func TestEvaluateGradMatchesCPU(t *testing.T) {
	for _, D := range []int{32, 128} {
		t.Run(fmt.Sprintf("D%d", D), func(t *testing.T) { checkGrad(t, D) })
	}
}

func checkGrad(t *testing.T, D int) {
	dev := device(t)
	const V, length, warm = 333, 40, 8
	d := model.Synthetic(V, D, 3000, int64(D))
	for i := range d.E {
		d.E[i] = roundHalf(d.E[i])
	}
	var gs []*neat.Genome
	for i := range 7 {
		gs = append(gs, model.Grown(int64(i+1), D, 30*i))
	}
	// One genome too big for the kernel in the middle: its rows are absent.
	big := model.NewGenome(99, D)
	for range MaxNodes {
		big.AddMemory()
	}
	gs = append(gs[:3], append([]*neat.Genome{big}, gs[3:]...)...)
	e, err := New(dev, d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	starts := []int{5, 800, 2000}
	bpb, over, fit, dO, dS, err := e.EvaluateGrad(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	gradBits := append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(&e.bits.Bytes()[0])), len(dS))...)
	if over != 1 || len(fit) != len(gs)-1 || !math.IsInf(bpb[3], 1) {
		t.Fatalf("over %d, fit %v, bpb %v", over, fit, bpb)
	}
	for _, i := range fit {
		if i == 3 {
			t.Fatal("the oversized genome has rows")
		}
	}
	T := length - warm
	if len(dO) != len(fit)*len(starts)*T*D || len(dS) != len(fit)*len(starts)*T {
		t.Fatalf("%d dO, %d dS", len(dO), len(dS))
	}
	// bpb is Evaluate's.
	want, _, err := e.Evaluate(gs, starts, length, warm)
	if err != nil {
		t.Fatal(err)
	}
	xentBits := unsafe.Slice((*float32)(unsafe.Pointer(&e.bits.Bytes()[0])), len(dS))
	for i := range gs {
		if math.IsInf(want[i], 1) {
			if !math.IsInf(bpb[i], 1) {
				t.Fatalf("genome %d: %g, want +Inf", i, bpb[i])
			}
		} else if math.Abs(bpb[i]-want[i]) > 2e-3*want[i] {
			t.Fatalf("genome %d: EvaluateGrad %g, Evaluate %g", i, bpb[i], want[i])
		}
	}
	prior := d.Prior()
	var worstBits, worstO, worstS, errO, sumSq float64
	nRows := 0
	for gi, i := range fit {
		s := float64(model.LogitScale(gs[i], D))
		for w, st := range starts {
			rows := ref.Rows(gs[i], d, d.Train, st, length, warm)
			for k, o := range rows {
				// The kernel reads the rows as halves.
				for q := range o {
					o[q] = roundHalf(o[q])
				}
				r := (gi*len(starts)+w)*T + k
				target := int(d.Train[st+warm+k+1])
				wb, wo, ws := gradRef(func(j int) []float32 { return d.Row(int32(j)) }, prior, V, o, s, target)
				worstBits = math.Max(worstBits, math.Abs(wb-float64(gradBits[r])))
				var num, den float64
				for k := range wo {
					g := float64(dO[r*D+k])
					if math.IsNaN(g) {
						t.Fatalf("row %d: NaN", r)
					}
					num += (g - wo[k]) * (g - wo[k])
					den += wo[k] * wo[k]
				}
				errO, sumSq, nRows = math.Max(errO, math.Sqrt(num)), sumSq+den, nRows+1
				worstS = math.Max(worstS, math.Abs(float64(dS[r])-ws)/(math.Abs(ws)+5e-2))
			}
		}
	}
	// A row whose target takes nearly all the probability has a gradient near
	// zero, (1-p) times something, while the rows the card reads differ from
	// the CPU's by netrun's rounding amplified by the scale: its error is not
	// small against its own norm. The error is therefore held to the root mean
	// square norm of dO over the batch.
	worstO = errO / math.Sqrt(sumSq/float64(nRows))
	worstX := 0.0
	for r := range gradBits {
		worstX = math.Max(worstX, math.Abs(float64(gradBits[r]-xentBits[r])))
	}
	t.Logf("fused bits against xent bits: worst %g", worstX)
	t.Logf("worst: bits %g, dO relative norm %g, dS relative %g", worstBits, worstO, worstS)
	// Against the CPU the rows differ by what netrun computes in float32 and
	// stores as halves, which the scale amplifies: the bits are held to xent's.
	if worstX > 1e-3 || worstBits > 0.1 || worstO > 5e-2 || worstS > 0.25 {
		t.Fatalf("worst: xent %g, bits %g, dO %g, dS %g", worstX, worstBits, worstO, worstS)
	}
}

// bpbRow reads the bits of row r of the last evaluation.
func bpbRow(e *Evaluator, r int) float32 {
	return *(*float32)(unsafe.Add(unsafe.Pointer(&e.bits.Bytes()[0]), 4*r))
}

func TestEvaluateGradRefusesTree(t *testing.T) {
	dev := device(t)
	d := model.Synthetic(100, 32, 1000, 3)
	tr := model.BuildTree(d)
	e, err := NewTree(dev, d, tr)
	if err != nil {
		t.Skip(err)
	}
	defer e.Close()
	if _, _, _, _, _, err := e.EvaluateGrad(nil, []int{0}, 40, 8); err == nil {
		t.Fatal("no error")
	}
}

// The kernel alone on random rows (the ones xent's test uses), where the CPU
// and the card read the same halves: the gradient is checked tightly. Vocab
// 300 is not a multiple of the 64 columns of a tile, rows 1000 not of the 64
// rows of a workgroup.
func TestXentGradKernel(t *testing.T) {
	for _, D := range []int{32, 64, 128} {
		t.Run(fmt.Sprintf("scalar/D%d", D), func(t *testing.T) { checkXentGrad(t, D, false) })
		t.Run(fmt.Sprintf("coop/D%d", D), func(t *testing.T) {
			if !device(t).Coopmat() {
				t.Skip("no cooperative matrices")
			}
			checkXentGrad(t, D, true)
		})
	}
}

func checkXentGrad(t *testing.T, D int, coop bool) {
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
	prior := d.Prior()
	up := func(b []byte) *vk.Buffer {
		buf, err := upload(dev, b, vk.UsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(buf.Close)
		return buf
	}
	local := func(n int) *vk.Buffer {
		b, err := dev.Local(4*n, vk.UsageStorage|vk.UsageTransferSrc)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Close)
		return b
	}
	genB, tokB, embB := up(words(gen.words)), up(words(toks)), up(words(packHalves(d.E)))
	rowsB, priorB := up(words(packHalves(rows))), up(floats(prior))
	bitsB, dOB, dSB := local(R), local(R*D), local(R)
	read := func(b *vk.Buffer, n int) []float32 {
		out, err := dev.Readback(4*n, vk.UsageTransferDst)
		if err != nil {
			t.Fatal(err)
		}
		defer out.Close()
		if err := dev.Submit(func(r *vk.Recorder) { r.Copy(out, 0, b, 4*n) }); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(&out.Bytes()[0])), n)...)
	}
	coopDim, tile := 0, xentGradRows
	if coop {
		coopDim, tile = D, xentGradCoopRows
	}
	pipe, err := newXentGradPipe(dev, coopDim, xentCoopWave)
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	set, err := pipe.NewSet([]*vk.Buffer{genB, tokB, embB, rowsB, bitsB, priorB, dOB, dSB})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	push := xentPush{R, V, uint32(D), scored, windows, warm, uint32(gen.startsOff), uint32(gen.scaleOff)}
	if err := dev.Submit(func(r *vk.Recorder) {
		if coop {
			r.DispatchWide(set, 1, uint32((R+tile-1)/tile), unsafe.Pointer(&push))
		} else {
			r.Dispatch(set, uint32((R+tile-1)/tile), unsafe.Pointer(&push))
		}
	}); err != nil {
		t.Fatal(err)
	}
	bits, dO, dS := read(bitsB, R), read(dOB, R*D), read(dSB, R)
	var worstB, worstO, worstS float64
	for r := range R {
		p := r / scored
		g, w := p/windows, p%windows
		target := int(d.Train[starts[w]+warm+r%scored+1])
		wb, wo, ws := gradRef(func(j int) []float32 { return d.Row(int32(j)) }, prior, V, rows[r*D:(r+1)*D], float64(scales[g]), target)
		worstB = math.Max(worstB, math.Abs(wb-float64(bits[r])))
		var num, den float64
		for k := range wo {
			x := float64(dO[r*D+k])
			num += (x - wo[k]) * (x - wo[k])
			den += wo[k] * wo[k]
		}
		worstO = math.Max(worstO, math.Sqrt(num)/(math.Sqrt(den)+1e-3))
		worstS = math.Max(worstS, math.Abs(float64(dS[r])-ws)/(math.Abs(ws)+1e-2))
	}
	t.Logf("worst: bits %g, dO relative norm %g, dS relative %g", worstB, worstO, worstS)
	// P goes through the matrix cores as halves: dO is held to 1e-2.
	if worstB > 1e-3 || worstO > 1e-2 || worstS > 1e-3 {
		t.Fatal("too far from the CPU")
	}
}
