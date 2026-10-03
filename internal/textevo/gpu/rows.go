package gpu

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

// genLayout is the per-generation upload: window starts, logit scale per
// genome, record offset per genome, the pairs each netrun variant runs, then
// the records.
type genLayout struct {
	words               []uint32
	startsOff, scaleOff int
	goffOff             int
	genomes             int
	// listOff is where the pairs (genome*windows+window) start: first the
	// pairs[0] of the small variant, then the pairs[1] of the big one.
	listOff int
	pairs   [2]int
	// tapeOff is where the table of learn.go starts, after the records.
	tapeOff int
}

// layoutSize is the number of words of the upload for these records.
func layoutSize(recs [][]uint32, starts int) int {
	total := starts + 2*len(recs) + len(recs)*starts
	for _, r := range recs {
		total += len(r)
	}
	return total
}

// layoutInto writes [starts K][scale G][goff G][pairs G*K][records] into
// dst, which holds layoutSize words. class gives the netrun variant of each
// record (see netClass). The record offsets are word indices into the whole
// upload, which is what the kernel reads. The records are copied in parallel.
func layoutInto(dst []uint32, recs [][]uint32, class []int8, scales []float32, starts []int) genLayout {
	g, k := len(recs), len(starts)
	l := genLayout{startsOff: 0, scaleOff: k, goffOff: k + g, listOff: k + 2*g, genomes: g}
	for i, s := range starts {
		dst[i] = uint32(s)
	}
	at := l.listOff + g*k
	offs := make([]int, g)
	for i, r := range recs {
		dst[l.scaleOff+i] = math.Float32bits(scales[i])
		dst[l.goffOff+i] = uint32(at)
		offs[i] = at
		at += len(r)
	}
	list := dst[l.listOff : l.listOff+g*k]
	n := 0
	for c := range int8(2) {
		for i := range g {
			if class[i] != c {
				continue
			}
			for w := range k {
				list[n] = uint32(i*k + w)
				n++
			}
			l.pairs[c] += k
		}
	}
	parallel(g, func(i int) { copy(dst[offs[i]:], recs[i]) })
	l.words = dst[:at]
	return l
}

// parallel runs f(i) for i in [0, n) on every CPU, in contiguous chunks.
func parallel(n int, f func(i int)) {
	workers := max(min(runtime.NumCPU(), n), 1)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += max(chunk, 1) {
		hi := min(lo+max(chunk, 1), n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				f(i)
			}
		}()
	}
	wg.Wait()
}

// buildGen packs flats, which must fit, and lays out the upload in a new
// slice.
func buildGen(flats []*neat.Flat, scales []float32, starts []int) genLayout {
	recs := make([][]uint32, len(flats))
	class := make([]int8, len(flats))
	parallel(len(flats), func(i int) {
		recs[i] = Pack(nil, flats[i])
		class[i] = int8(netClass(flats[i]))
	})
	return layoutInto(make([]uint32, layoutSize(recs, len(starts))), recs, class, scales, starts)
}

// netPush is the push constant block of netrun.comp.
type netPush struct {
	Pairs, Windows, Len, Warm, Dim, StartsOff, GoffOff, ListOff, Outs uint32
	// TapeOff is read by the tape variants only (see learn.go).
	TapeOff uint32
}

// newNetPipes builds the small and the big variant of netrun.
func newNetPipes(d *vk.Device) ([2]*vk.Pipeline, error) {
	var p [2]*vk.Pipeline
	var err error
	for i, spv := range [][]byte{netrunSPV, netrunBigSPV} {
		if p[i], err = d.NewPipeline(spv, 4, 9*4); err != nil {
			for _, q := range p {
				if q != nil {
					q.Close()
				}
			}
			return [2]*vk.Pipeline{}, err
		}
	}
	return p, nil
}

// recordNet records the netrun dispatches of a generation, one per variant
// with work. They write different rows, so they may overlap. outs is the
// number of prediction outputs of the networks.
func recordNet(r *vk.Recorder, sets [2]*vk.Set, gen genLayout, windows, length, warm, dim, outs int) {
	off := gen.listOff
	for c, n := range gen.pairs {
		if n == 0 {
			continue
		}
		push := netPush{uint32(n), uint32(windows), uint32(length), uint32(warm), uint32(dim),
			uint32(gen.startsOff), uint32(gen.goffOff), uint32(off), uint32(outs), uint32(gen.tapeOff)}
		r.Dispatch(sets[c], uint32(n), unsafe.Pointer(&push))
		off += n
	}
}

// upload copies data into a new device-local buffer through a host one.
func upload(d *vk.Device, data []byte, usage uint32) (*vk.Buffer, error) {
	dst, err := d.Local(len(data), usage|vk.UsageTransferDst)
	if err != nil {
		return nil, err
	}
	src, err := d.Host(len(data), vk.UsageTransferSrc)
	if err != nil {
		dst.Close()
		return nil, err
	}
	defer src.Close()
	copy(src.Bytes(), data)
	if err := d.Submit(func(r *vk.Recorder) { r.Copy(dst, 0, src, len(data)) }); err != nil {
		dst.Close()
		return nil, err
	}
	return dst, nil
}
