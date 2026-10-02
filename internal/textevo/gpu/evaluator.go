package gpu

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

// maxGroups is the Vulkan limit on the workgroups of one dispatch dimension.
const maxGroups = 65535

// Evaluator scores whole generations on the card: one submission runs every
// genome on every window, the embedding and the corpus stay resident.
type Evaluator struct {
	d    *vk.Device
	data *prep.Data

	netPipe, xentPipe *vk.Pipeline
	// Resident inputs. Train and validation tokens differ, the embedding is
	// shared, in float for netrun and in halves for xent.
	tokens       [2]*vk.Buffer
	emb32, emb16 *vk.Buffer

	// Growable per-generation buffers, and the sets that bind them, one pair
	// of sets for each token buffer.
	up, gen, rows, bits *vk.Buffer
	capUp, capGen       int
	capRows, capBits    int
	netSet, xentSet     [2]*vk.Set

	// Timing is how the last Evaluate or Validate spent its time: packing the
	// genomes on the CPU, the upload and the kernels, summing the bits.
	Timing struct{ Pack, GPU, Sum time.Duration }

	// skipNet and skipXent leave a kernel out of the submission, so that a
	// benchmark can time the other one alone. Tests only.
	skipNet, skipXent bool
	// coop says that xent runs on the matrix cores.
	coop bool
}

// New uploads the corpus and the embedding of data and builds the kernels.
func New(d *vk.Device, data *prep.Data) (*Evaluator, error) {
	e := &Evaluator{d: d, data: data}
	ok := false
	defer func() {
		if !ok {
			e.Close()
		}
	}()
	if data.Dim%2 != 0 {
		return nil, fmt.Errorf("gpu: the embedding dimension %d must be even", data.Dim)
	}
	var err error
	for i, ids := range [2][]int32{data.Train, data.Val} {
		w := make([]uint32, max(len(ids), 1))
		for j, x := range ids {
			w[j] = uint32(x)
		}
		if e.tokens[i], err = upload(d, words(w), vk.UsageStorage); err != nil {
			return nil, err
		}
	}
	if e.emb32, err = upload(d, floats(data.E), vk.UsageStorage); err != nil {
		return nil, err
	}
	if e.emb16, err = upload(d, words(packHalves(data.E)), vk.UsageStorage); err != nil {
		return nil, err
	}
	if e.netPipe, err = d.NewPipeline(netrunSPV, 4, 7*4); err != nil {
		return nil, err
	}
	// The matrix cores take xent when the device has them and the embedding is
	// the width the kernel was built for; NEATTEXT_SCALAR_XENT forces the
	// scalar kernel, to compare the two.
	spv := xentSPV
	if d.Coopmat() && data.Dim == xentCoopDim && os.Getenv("NEATTEXT_SCALAR_XENT") == "" {
		spv = xentCoopSPV
		e.coop = true
	}
	if e.xentPipe, err = d.NewPipeline(spv, 5, 8*4); err != nil {
		return nil, err
	}
	ok = true
	return e, nil
}

func words(w []uint32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w))
}

func floats(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), 4*len(f))
}

// packHalves rounds floats to halves and packs two to a word, low half first.
func packHalves(x []float32) []uint32 {
	w := make([]uint32, (len(x)+1)/2)
	for i := range w {
		w[i] = uint32(floatToHalf(x[2*i]))
		if 2*i+1 < len(x) {
			w[i] |= uint32(floatToHalf(x[2*i+1])) << 16
		}
	}
	return w
}

// Close releases the sets, the pipelines and every buffer.
func (e *Evaluator) Close() {
	e.dropSets()
	for _, p := range []*vk.Pipeline{e.netPipe, e.xentPipe} {
		if p != nil {
			p.Close()
		}
	}
	e.netPipe, e.xentPipe = nil, nil
	for _, b := range []*vk.Buffer{e.tokens[0], e.tokens[1], e.emb32, e.emb16, e.up, e.gen, e.rows, e.bits} {
		if b != nil {
			b.Close()
		}
	}
	e.tokens, e.emb32, e.emb16 = [2]*vk.Buffer{}, nil, nil
	e.up, e.gen, e.rows, e.bits = nil, nil, nil, nil
}

func (e *Evaluator) dropSets() {
	for i := range e.netSet {
		if e.netSet[i] != nil {
			e.netSet[i].Close()
			e.netSet[i] = nil
		}
		if e.xentSet[i] != nil {
			e.xentSet[i].Close()
			e.xentSet[i] = nil
		}
	}
}

// grow makes the per-generation buffers at least as large as asked, doubling
// so that a population that slowly gets bigger reallocates rarely, and
// rebuilds the sets that point at them.
func (e *Evaluator) grow(upBytes, genBytes, rowBytes, bitBytes int) error {
	if upBytes <= e.capUp && genBytes <= e.capGen && rowBytes <= e.capRows && bitBytes <= e.capBits {
		return nil
	}
	e.dropSets()
	for _, c := range []struct {
		b    **vk.Buffer
		cap  *int
		need int
		make func(int) (*vk.Buffer, error)
	}{
		{&e.up, &e.capUp, upBytes, func(n int) (*vk.Buffer, error) { return e.d.Host(n, vk.UsageTransferSrc) }},
		{&e.gen, &e.capGen, genBytes, func(n int) (*vk.Buffer, error) {
			return e.d.Local(n, vk.UsageStorage|vk.UsageTransferDst)
		}},
		{&e.rows, &e.capRows, rowBytes, func(n int) (*vk.Buffer, error) { return e.d.Local(n, vk.UsageStorage) }},
		{&e.bits, &e.capBits, bitBytes, func(n int) (*vk.Buffer, error) { return e.d.Readback(n, vk.UsageStorage) }},
	} {
		if c.need <= *c.cap {
			continue
		}
		n := max(c.need, 2**c.cap)
		if *c.b != nil {
			(*c.b).Close()
			*c.b, *c.cap = nil, 0
		}
		b, err := c.make(n)
		if err != nil {
			return err
		}
		*c.b, *c.cap = b, n
	}
	for i := range e.netSet {
		var err error
		if e.netSet[i], err = e.netPipe.NewSet([]*vk.Buffer{e.gen, e.tokens[i], e.emb32, e.rows}); err != nil {
			return err
		}
		if e.xentSet[i], err = e.xentPipe.NewSet([]*vk.Buffer{e.gen, e.tokens[i], e.emb16, e.rows, e.bits}); err != nil {
			return err
		}
	}
	return nil
}

// Evaluate returns the bits per byte of each genome on the train windows
// at starts, +Inf for a genome too big for the kernel. Oversized counts them.
func (e *Evaluator) Evaluate(gs []*neat.Genome, starts []int, length, warm int) (bpb []float64, oversized int, err error) {
	return e.evaluate(gs, 0, e.data.Train, starts, length, warm)
}

// Validate scores one genome on the fixed validation windows.
func (e *Evaluator) Validate(g *neat.Genome) (float64, error) {
	ids := e.data.Val
	bpb, over, err := e.evaluate([]*neat.Genome{g}, 1, ids, model.ValStarts(len(ids)), model.ValLen, model.Warm)
	if err != nil {
		return 0, err
	}
	if over > 0 {
		return math.Inf(1), nil
	}
	return bpb[0], nil
}

func (e *Evaluator) evaluate(gs []*neat.Genome, kind int, ids []int32, starts []int, length, warm int) ([]float64, int, error) {
	d := e.data
	start := time.Now()
	flats := make([]*neat.Flat, len(gs))
	var wg sync.WaitGroup
	next := make(chan int)
	for range min(runtime.NumCPU(), max(len(gs), 1)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				if f := gs[i].BuildNetwork().Flat(); Fits(f) {
					flats[i] = f
				}
			}
		}()
	}
	for i := range gs {
		next <- i
	}
	close(next)
	wg.Wait()
	bpb := make([]float64, len(gs))
	var fit []*neat.Flat
	var fitIdx []int
	var scales []float32
	over := 0
	for i, f := range flats {
		if f == nil {
			bpb[i] = math.Inf(1)
			over++
			continue
		}
		fit = append(fit, f)
		fitIdx = append(fitIdx, i)
		scales = append(scales, model.LogitScale(gs[i], d.Dim))
	}
	if len(fit) == 0 {
		e.Timing.Pack, e.Timing.GPU, e.Timing.Sum = time.Since(start), 0, 0
		return bpb, over, nil
	}
	gen := buildGen(fit, scales, starts)
	e.Timing.Pack = time.Since(start)

	scored := length - warm
	pairs := len(fit) * len(starts)
	rows := pairs * scored
	if pairs > maxGroups {
		return nil, 0, fmt.Errorf("gpu: %d genomes x %d windows is %d pairs, over the %d workgroups of one dispatch: lower -pop or -windows",
			len(fit), len(starts), pairs, maxGroups)
	}
	if (rows+63)/64 > maxGroups {
		return nil, 0, fmt.Errorf("gpu: %d rows to score, over what one dispatch of xent covers (%d): lower -pop, -windows or -len",
			rows, 64*maxGroups)
	}

	start = time.Now()
	genBytes := 4 * len(gen.words)
	if err := e.grow(genBytes, genBytes, rows*d.Dim*2, rows*4); err != nil {
		return nil, 0, err
	}
	copy(e.up.Bytes(), words(gen.words))
	netPush := [7]uint32{uint32(pairs), uint32(len(starts)), uint32(length), uint32(warm), uint32(d.Dim),
		uint32(gen.startsOff), uint32(gen.goffOff)}
	xp := xentPush{uint32(rows), uint32(d.Vocab()), uint32(d.Dim), uint32(scored), uint32(len(starts)), uint32(warm),
		uint32(gen.startsOff), uint32(gen.scaleOff)}
	err := e.d.Submit(func(r *vk.Recorder) {
		r.Copy(e.gen, 0, e.up, genBytes)
		r.TransferBarrier()
		if !e.skipNet {
			r.Dispatch(e.netSet[kind], uint32(pairs), unsafe.Pointer(&netPush))
			r.Barrier()
		}
		if !e.skipXent {
			r.Dispatch(e.xentSet[kind], uint32((rows+63)/64), unsafe.Pointer(&xp))
		}
	})
	e.Timing.GPU = time.Since(start)
	if err != nil {
		return nil, 0, err
	}

	start = time.Now()
	bits := unsafe.Slice((*float32)(unsafe.Pointer(&e.bits.Bytes()[0])), rows)
	per := len(starts) * scored
	denom := float64(model.WindowBytes(d, ids, starts, length, warm))
	for j, i := range fitIdx {
		sum := 0.0
		for _, b := range bits[j*per : (j+1)*per] {
			sum += float64(b)
		}
		bpb[i] = sum / denom
	}
	e.Timing.Sum = time.Since(start)
	return bpb, over, nil
}
