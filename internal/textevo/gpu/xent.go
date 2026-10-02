package gpu

import (
	_ "embed"
	"fmt"

	"github.com/ThiraSoft/golem/vk"
)

//go:embed xent.spv
var xentSPV []byte

//go:embed xent_coop.spv
var xentCoop128SPV []byte

//go:embed xent_coop64.spv
var xentCoop64SPV []byte

//go:embed xent_coop32.spv
var xentCoop32SPV []byte

// xentCoopSPV holds xent_coop.comp built for each embedding width a run may
// prepare: the scalar kernel is ten to twenty times slower, which made it
// most of a generation at the smaller widths.
var xentCoopSPV = map[int][]byte{32: xentCoop32SPV, 64: xentCoop64SPV, 128: xentCoop128SPV}

// xentPush is the push constant block of xent.comp.
type xentPush struct {
	Rows, Vocab, Dim, Scored, Windows, Warm, StartsOff, ScaleOff uint32
}

// xentCoopRows is how many rows one workgroup of xent_coop.comp covers.
const xentCoopRows = 128

// xentCoopWave is the wave width xent_coop.comp divides its rows for.
const xentCoopWave = 64

// newXentPipe builds the scalar xent kernel and, when coopDim is not zero, the
// matrix one for that embedding width as its binary of width one at the given
// wave width, so that a dispatch
// with DispatchWide(set, 1, ...) runs it over the same sets. The wave is
// asked for rather than taken: RADV runs compute at sixty-four by default but
// may offer thirty-two, and xent_coop.comp refuses any width but its own.
func newXentPipe(d *vk.Device, coopDim int, wave uint32) (*vk.Pipeline, error) {
	p, err := d.NewPipeline(xentSPV, 6, 8*4)
	if err != nil || coopDim == 0 {
		return p, err
	}
	spv, ok := xentCoopSPV[coopDim]
	if !ok {
		p.Close()
		return nil, fmt.Errorf("gpu: xent_coop is not built for dimension %d", coopDim)
	}
	if err := p.WideWave(1, spv, wave); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}
