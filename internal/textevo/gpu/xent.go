package gpu

import (
	_ "embed"

	"github.com/ThiraSoft/golem/vk"
)

//go:embed xent.spv
var xentSPV []byte

//go:embed xent_coop.spv
var xentCoopSPV []byte

// xentPush is the push constant block of xent.comp.
type xentPush struct {
	Rows, Vocab, Dim, Scored, Windows, Warm, StartsOff, ScaleOff uint32
}

// xentCoopDim is the embedding width xent_coop.comp is built for.
const xentCoopDim = 128

// xentCoopRows is how many rows one workgroup of xent_coop.comp covers.
const xentCoopRows = 128

// xentCoopWave is the wave width xent_coop.comp divides its rows for.
const xentCoopWave = 64

// newXentPipe builds the scalar xent kernel and, with coop, the matrix one
// as its binary of width one at the given wave width, so that a dispatch
// with DispatchWide(set, 1, ...) runs it over the same sets. The wave is
// asked for rather than taken: RADV runs compute at sixty-four by default but
// may offer thirty-two, and xent_coop.comp refuses any width but its own.
func newXentPipe(d *vk.Device, coop bool, wave uint32) (*vk.Pipeline, error) {
	p, err := d.NewPipeline(xentSPV, 5, 8*4)
	if err != nil || !coop {
		return p, err
	}
	if err := p.WideWave(1, xentCoopSPV, wave); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}
