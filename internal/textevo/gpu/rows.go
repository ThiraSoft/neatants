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

// buildGen lays out [starts K][scale G][goff G][records]. The record offsets
// are word indices into the whole upload, which is what the kernel reads.
func buildGen(flats []*neat.Flat, scales []float32, starts []int) genLayout {
	g := len(flats)
	recs := make([][]uint32, g)
	workers := runtime.NumCPU()
	chunk := (g + workers - 1) / max(workers, 1)
	var wg sync.WaitGroup
	for lo := 0; lo < g; lo += max(chunk, 1) {
		hi := min(lo+max(chunk, 1), g)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				recs[i] = Pack(nil, flats[i])
			}
		}()
	}
	wg.Wait()
	l := genLayout{startsOff: 0, scaleOff: len(starts), goffOff: len(starts) + g, genomes: g}
	total := len(starts) + 2*g
	for _, r := range recs {
		total += len(r)
	}
	l.words = make([]uint32, len(starts)+2*g, total)
	for i, s := range starts {
		l.words[i] = uint32(s)
	}
	at := len(starts) + 2*g
	for i, r := range recs {
		l.words[l.scaleOff+i] = math.Float32bits(scales[i])
		l.words[l.goffOff+i] = uint32(at)
		at += len(r)
		l.words = append(l.words, r...)
	}
	return l
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
