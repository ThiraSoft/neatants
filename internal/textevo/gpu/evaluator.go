package gpu

import (
	"fmt"
	"math"
	"os"
	"slices"
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

	// netPipe holds the small and the big variant of netrun.
	netPipe  [2]*vk.Pipeline
	xentPipe *vk.Pipeline
	// Resident inputs. Train and validation tokens differ, the embedding is
	// shared, in float for netrun and in halves for xent.
	tokens       [2]*vk.Buffer
	emb32, emb16 *vk.Buffer
	// prior is the unigram log-prior every logit starts from.
	prior *vk.Buffer

	// Growable per-generation buffers, and the sets that bind them, one pair
	// of sets for each token buffer.
	up, gen, rows, bits *vk.Buffer
	capUp, capGen       int
	capRows, capBits    int
	netSet              [2][2]*vk.Set // [tokens][variant]
	xentSet             [2]*vk.Set

	// Timing is how the last Evaluate or Validate spent its time: packing the
	// genomes on the CPU, the upload and the kernels, summing the bits.
	Timing struct{ Pack, GPU, Sum time.Duration }

	// skipNet and skipXent leave a kernel out of the submission, so that a
	// benchmark can time the other one alone. Tests only.
	skipNet, skipXent bool
	// coop says that xent runs on the matrix cores.
	coop bool

	// slots are the records of the last two generations, flip says which set
	// the last one used, and prev finds its records by lineage.
	slots [2][]packed
	flip  int
	prev  map[int]*packed
	// xentRows is how many rows one workgroup of xent covers.
	xentRows int
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
	if e.prior, err = upload(d, floats(data.Prior()), vk.UsageStorage); err != nil {
		return nil, err
	}
	if e.netPipe, err = newNetPipes(d); err != nil {
		return nil, err
	}
	// The matrix cores take xent when the device has them and the kernel was
	// built for the embedding's width; NEATTEXT_SCALAR_XENT forces the
	// scalar kernel, to compare the two.
	_, built := xentCoopSPV[data.Dim]
	e.coop = d.Coopmat() && built && os.Getenv("NEATTEXT_SCALAR_XENT") == ""
	e.xentRows = 64
	coopDim := 0
	if e.coop {
		e.xentRows = xentCoopRows
		coopDim = data.Dim
	}
	if e.xentPipe, err = newXentPipe(d, coopDim, xentCoopWave); err != nil {
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
	for _, p := range []*vk.Pipeline{e.netPipe[0], e.netPipe[1], e.xentPipe} {
		if p != nil {
			p.Close()
		}
	}
	e.netPipe, e.xentPipe = [2]*vk.Pipeline{}, nil
	for _, b := range []*vk.Buffer{e.tokens[0], e.tokens[1], e.emb32, e.emb16, e.prior, e.up, e.gen, e.rows, e.bits} {
		if b != nil {
			b.Close()
		}
	}
	e.tokens, e.emb32, e.emb16, e.prior = [2]*vk.Buffer{}, nil, nil, nil
	e.up, e.gen, e.rows, e.bits = nil, nil, nil, nil
}

func (e *Evaluator) dropSets() {
	for i := range e.netSet {
		for c := range e.netSet[i] {
			if e.netSet[i][c] != nil {
				e.netSet[i][c].Close()
				e.netSet[i][c] = nil
			}
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
		for c := range e.netSet[i] {
			if e.netSet[i][c], err = e.netPipe[c].NewSet([]*vk.Buffer{e.gen, e.tokens[i], e.emb32, e.rows}); err != nil {
				return err
			}
		}
		if e.xentSet[i], err = e.xentPipe.NewSet([]*vk.Buffer{e.gen, e.tokens[i], e.emb16, e.rows, e.bits, e.prior}); err != nil {
			return err
		}
	}
	return nil
}

// Evaluate returns the bits per byte of each genome on the train windows
// at starts, +Inf for a genome too big for the kernel. Oversized counts them.
// The genomes must not be changed in place afterwards: the next call compares
// copies of them with them to reuse their records (copies may be changed).
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
	slots := e.pack(gs, kind == 0)
	bpb := make([]float64, len(gs))
	var recs [][]uint32
	var class []int8
	var fitIdx []int
	var scales []float32
	over := 0
	for i := range slots {
		if !slots[i].fits {
			bpb[i] = math.Inf(1)
			over++
			continue
		}
		recs = append(recs, slots[i].rec)
		class = append(class, slots[i].class)
		fitIdx = append(fitIdx, i)
		scales = append(scales, model.LogitScale(gs[i], d.Dim))
	}
	if len(recs) == 0 {
		e.Timing.Pack, e.Timing.GPU, e.Timing.Sum = time.Since(start), 0, 0
		return bpb, over, nil
	}

	scored := length - warm
	pairs := len(recs) * len(starts)
	rows := pairs * scored
	if pairs > maxGroups {
		return nil, 0, fmt.Errorf("gpu: %d genomes x %d windows is %d pairs, over the %d workgroups of one dispatch: lower -pop or -windows",
			len(recs), len(starts), pairs, maxGroups)
	}
	if (rows+e.xentRows-1)/e.xentRows > maxGroups {
		return nil, 0, fmt.Errorf("gpu: %d rows to score, over what one dispatch of xent covers (%d): lower -pop, -windows or -len",
			rows, e.xentRows*maxGroups)
	}

	genBytes := 4 * layoutSize(recs, len(starts))
	if err := e.grow(genBytes, genBytes, rows*d.Dim*2, rows*4); err != nil {
		return nil, 0, err
	}
	// The records go straight into the staging buffer.
	up := unsafe.Slice((*uint32)(unsafe.Pointer(&e.up.Bytes()[0])), genBytes/4)
	gen := layoutInto(up, recs, class, scales, starts)
	e.Timing.Pack = time.Since(start)
	start = time.Now()
	xp := xentPush{uint32(rows), uint32(d.Vocab()), uint32(d.Dim), uint32(scored), uint32(len(starts)), uint32(warm),
		uint32(gen.startsOff), uint32(gen.scaleOff)}
	err := e.d.Submit(func(r *vk.Recorder) {
		r.Copy(e.gen, 0, e.up, genBytes)
		r.TransferBarrier()
		if !e.skipNet {
			recordNet(r, e.netSet[kind], gen, len(starts), length, warm, d.Dim)
			r.Barrier()
		}
		if !e.skipXent {
			groups := uint32((rows + e.xentRows - 1) / e.xentRows)
			if e.coop {
				r.DispatchWide(e.xentSet[kind], 1, groups, unsafe.Pointer(&xp))
			} else {
				r.Dispatch(e.xentSet[kind], groups, unsafe.Pointer(&xp))
			}
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
		if math.IsNaN(sum) {
			// Only xent_coop writes NaN, when it runs at a wave width it
			// was not written for; the scores would be meaningless.
			return nil, 0, fmt.Errorf("gpu: xent returned NaN for genome %d: the matrix kernel ran at the wrong wave width, set NEATTEXT_SCALAR_XENT=1", i)
		}
		bpb[i] = sum / denom
	}
	e.Timing.Sum = time.Since(start)
	return bpb, over, nil
}

// packed is the record of one genome, ready for the upload, or the fact that
// it does not fit the kernel, with the genome it was built from. Keeping the
// genome rather than a copy of its genes saves copying some hundred kilobytes
// a genome every generation for a record only champions reuse.
type packed struct {
	rec   []uint32
	fits  bool
	class int8
	g     *neat.Genome
}

// same reports whether g has exactly the genes p was built from. Origin only
// says what a genome was copied from: a caller may change the genes of a copy
// without Mutate, and a stale record would then score another network.
func (p *packed) same(g *neat.Genome) bool {
	o := p.g
	return o != nil && o.NumInputs == g.NumInputs && o.NumOutputs == g.NumOutputs &&
		slices.Equal(o.Nodes, g.Nodes) && slices.Equal(o.Conns, g.Conns)
}

// pack builds the record of every genome. With cache set, an unchanged copy
// of a genome of the previous generation (a champion carried over, found by
// Origin and checked gene by gene) reuses that genome's record; the slots alternate between two sets so that the records
// a copy reads are never the ones being rewritten, and their buffers are
// reused from one generation to the next instead of being allocated again.
func (e *Evaluator) pack(gs []*neat.Genome, cache bool) []packed {
	var slots []packed
	var prev map[int]*packed
	if cache {
		e.flip ^= 1
		if len(e.slots[e.flip]) < len(gs) {
			e.slots[e.flip] = append(e.slots[e.flip], make([]packed, len(gs)-len(e.slots[e.flip]))...)
		}
		slots, prev = e.slots[e.flip][:len(gs)], e.prev
	} else {
		slots = make([]packed, len(gs))
	}
	parallel(len(gs), func(i int) {
		g, s := gs[i], &slots[i]
		if cache {
			s.g = g
		}
		if p := prev[g.Origin]; g.Origin > 0 && p != nil && p.same(g) {
			s.fits, s.class, s.rec = p.fits, p.class, append(s.rec[:0], p.rec...)
			return
		}
		f := g.BuildNetwork().Flat()
		s.class = int8(netClass(f))
		if s.fits = s.class >= 0; s.fits {
			s.rec = Pack(s.rec[:0], f)
		} else {
			s.rec = s.rec[:0]
		}
	})
	if cache {
		e.prev = make(map[int]*packed, len(gs))
		for i, g := range gs {
			e.prev[g.Lineage()] = &slots[i]
		}
	}
	return slots
}
