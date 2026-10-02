package gpubrain

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

const savePath = "../../saves/colonies.json"

// benchGenomes returns genomes for the benchmark: the evolved ones of the
// real save when it is there (their size and depth are what the kernel meets
// in a run), else big grown ones of about 500 nodes. The string says which.
func benchGenomes() ([]*neat.Genome, string) {
	var save struct {
		Colonies []struct {
			Hall []*neat.Genome `json:"hall"`
		} `json:"colonies"`
	}
	if data, err := os.ReadFile(savePath); err == nil && json.Unmarshal(data, &save) == nil {
		var gs []*neat.Genome
		for _, c := range save.Colonies {
			for _, g := range c.Hall {
				if g != nil && g.NumInputs == 67 && g.NumOutputs == 8 && fits(g.BuildNetwork().Flat()) {
					gs = append(gs, g)
				}
			}
		}
		if len(gs) > 0 {
			return gs, fmt.Sprintf("%d genomes of the real save", len(gs))
		}
	}
	var gs []*neat.Genome
	for i := range 16 {
		gs = append(gs, grown(int64(i), 6000))
	}
	return gs, "grown genomes (no save)"
}

// BenchmarkDispatch times one dispatch of n networks that are already on the
// card, which is the steady state of a run. ms/dispatch is the wall clock of
// Begin, Start and Finish; gpu-ms-med and gpu-ms-min are the card's own time
// for the dispatch alone, which stays steady when another program shares the
// card, so they are the figures to compare kernels with.
func BenchmarkDispatch(b *testing.B) {
	gs, src := benchGenomes()
	b.Logf("networks from %s", src)
	for _, n := range []int{500, 2000, 5000, 10000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			d, err := vk.Open()
			if err != nil {
				b.Skipf("no Vulkan device: %v", err)
			}
			defer d.Close()
			bt, err := New(d, 72, 8, 3, n)
			if err != nil {
				b.Fatal(err)
			}
			defer bt.Close()
			nets := make([]*neat.Network, n)
			for i := range nets {
				nets[i] = gs[i%len(gs)].BuildNetwork()
			}
			in := make([]float64, 67)
			r := rand.New(rand.NewSource(1))
			// Set costs the CPU, not the card, so it runs once. The loop
			// re-opens the round (which forgets the births, or Start would
			// upload every network again) and times Start and Finish on the
			// same requests.
			bt.Begin(n)
			for i, net := range nets {
				for k := range in {
					in[k] = r.Float64()*2 - 1
				}
				bt.Set(i, net, in)
			}
			for range 2 { // the first round uploads the networks
				if err := bt.Start(); err != nil {
					b.Fatal(err)
				}
				if err := bt.Finish(); err != nil {
					b.Fatal(err)
				}
			}
			tl, err := d.NewTimeline(2)
			if err != nil {
				b.Fatal(err)
			}
			defer tl.Close()
			push := [3]uint32{uint32(n), 72, 8}
			var gpu []float64 // card-side time of the dispatch alone, in ms
			b.ResetTimer()
			for range b.N {
				bt.Begin(n) // forgets the births, keeps the requests
				if err := bt.Start(); err != nil {
					b.Fatal(err)
				}
				if err := bt.Finish(); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			// The wall clock above includes waiting for the card's turn when
			// something else (a game) uses it, which is noise for kernel work.
			// The card's own clock around the same dispatch does not, and the
			// median and the minimum of it are what to compare. AMD cards
			// tick this clock at 100 MHz.
			for range 40 {
				err := d.Start(func(r *vk.Recorder) {
					tl.Reset(r)
					tl.Stamp(r, "start")
					r.Dispatch(bt.set, uint32(n), unsafe.Pointer(&push))
					tl.Stamp(r, "dispatch")
				})
				if err == nil {
					err = d.Wait()
				}
				if err != nil {
					b.Fatal(err)
				}
				spans, err := tl.Spans()
				if err != nil {
					b.Fatal(err)
				}
				gpu = append(gpu, float64(spans[0].Ticks)/1e5)
			}
			slices.Sort(gpu)
			b.ReportMetric(gpu[len(gpu)/2], "gpu-ms-med")
			b.ReportMetric(gpu[0], "gpu-ms-min")
			per := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
			b.ReportMetric(per/float64(n), "ns/net")
			b.ReportMetric(per/1e6, "ms/dispatch")
		})
	}
}

// GPUBRAIN_SPV names a compiled kernel to benchmark instead of the embedded
// one, so variants can be compared without touching the package.
func init() {
	if p := os.Getenv("GPUBRAIN_SPV"); p != "" {
		activateSPV, _ = os.ReadFile(p)
	}
}
