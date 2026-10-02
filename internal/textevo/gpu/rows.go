package gpu

import (
	"math"
	"runtime"
	"sync"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

// genLayout is the per-generation upload: window starts, logit scale per
// genome, record offset per genome, then the records.
type genLayout struct {
	words               []uint32
	startsOff, scaleOff int
	goffOff             int
	genomes             int
}

// layoutSize is the number of words of the upload for these records.
func layoutSize(recs [][]uint32, starts int) int {
	total := starts + 2*len(recs)
	for _, r := range recs {
		total += len(r)
	}
	return total
}

// layoutInto writes [starts K][scale G][goff G][records] into dst, which holds
// layoutSize words. The record offsets are word indices into the whole
// upload, which is what the kernel reads. The records are copied in parallel.
func layoutInto(dst []uint32, recs [][]uint32, scales []float32, starts []int) genLayout {
	g := len(recs)
	l := genLayout{startsOff: 0, scaleOff: len(starts), goffOff: len(starts) + g, genomes: g}
	for i, s := range starts {
		dst[i] = uint32(s)
	}
	at := len(starts) + 2*g
	offs := make([]int, g)
	for i, r := range recs {
		dst[l.scaleOff+i] = math.Float32bits(scales[i])
		dst[l.goffOff+i] = uint32(at)
		offs[i] = at
		at += len(r)
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

// buildGen packs flats and lays out the upload in a new slice.
func buildGen(flats []*neat.Flat, scales []float32, starts []int) genLayout {
	recs := make([][]uint32, len(flats))
	parallel(len(flats), func(i int) { recs[i] = Pack(nil, flats[i]) })
	return layoutInto(make([]uint32, layoutSize(recs, len(starts))), recs, scales, starts)
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
