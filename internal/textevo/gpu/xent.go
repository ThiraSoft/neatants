package gpu

import _ "embed"

//go:embed xent.spv
var xentSPV []byte

// xentPush is the push constant block of xent.comp.
type xentPush struct {
	Rows, Vocab, Dim, Scored, Windows, Warm, StartsOff, ScaleOff uint32
}
