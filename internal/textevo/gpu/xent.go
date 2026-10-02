package gpu

import _ "embed"

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
