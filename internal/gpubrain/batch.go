package gpubrain

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DMAX_NODES=1024u -DMAX_EDGES=4096u -DMAX_PLASTIC=256u activate.comp -o activate.spv

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
	"github.com/ThiraSoft/neatants/neat"
)

//go:embed activate.spv
var activateSPV []byte

const (
	usageArena = vk.UsageStorage | vk.UsageTransferDst | vk.UsageTransferSrc
	slotBytes  = SlotWords * 4
	noSlot     = math.MaxUint32 // request slot the kernel skips
	stageStart = 4              // slots the staging buffer holds at first
)

// Batch evaluates many networks per dispatch. Each living network keeps a
// slot in a device-local arena from its first evaluation until it stops
// being asked for (ttl rounds): its weights learn and its cells remember on
// the card, and only the inputs and the outputs cross the bus each round.
//
// A round is Begin, Set for each request (from any goroutines), Start, then
// Finish once the caller has done something else, then Out.
type Batch struct {
	d                   *vk.Device
	pipe                *vk.Pipeline
	set                 *vk.Set
	arena               *vk.Buffer // Local, capacity*SlotWords words
	prev                *vk.Buffer // arena before this round's growth, copied into arena by Start
	prevSize            int
	req, in             *vk.Buffer // Host: slot per request, inputs
	out                 *vk.Buffer // Readback
	staging             *vk.Buffer // Host: packed newborns of this round
	inStride, outStride int
	ttl, capacity       int
	reqCap, stageCap    int

	mu       sync.Mutex
	free     []int32
	owner    []*neat.Network
	lastSeen []int
	round    int
	births   []birth     // newborns of this round: slot and staging index
	lastBorn int         // births of the previous round
	cpuOut   [][]float32 // per request, set for networks run on the CPU
	n        int         // requests this round

	uploaded, freed, onCPU int
}

type birth struct{ slot, staged int32 }

// New makes an evaluator for networks of at most inStride inputs and
// outStride outputs. ttl is how many calls a network may skip before its
// slot is freed (think_every + 1). capacity is the initial number of slots.
func New(d *vk.Device, inStride, outStride, ttl, capacity int) (*Batch, error) {
	capacity = max(capacity, 1)
	b := &Batch{d: d, inStride: inStride, outStride: outStride, ttl: ttl, reqCap: capacity, stageCap: stageStart}
	var err error
	if b.pipe, err = d.NewPipeline(activateSPV, 4, 12); err != nil {
		return nil, err
	}
	if b.arena, err = d.Local(capacity*slotBytes, usageArena); err != nil {
		b.Close()
		return nil, err
	}
	if b.staging, err = d.Host(stageStart*slotBytes, vk.UsageTransferSrc); err != nil {
		b.Close()
		return nil, err
	}
	if err = b.allocRequests(capacity); err != nil {
		b.Close()
		return nil, err
	}
	b.addSlots(0, capacity)
	return b, nil
}

// addSlots makes slots [from, to) available. They are pushed highest first
// so that pop hands out the lowest one.
func (b *Batch) addSlots(from, to int) {
	b.owner = append(b.owner, make([]*neat.Network, to-from)...)
	b.lastSeen = append(b.lastSeen, make([]int, to-from)...)
	for s := to - 1; s >= from; s-- {
		b.free = append(b.free, int32(s))
	}
	b.capacity = to
}

// allocRequests (re)creates the per-request buffers for n requests and
// rebuilds the set around them. Only called while nothing is in flight.
func (b *Batch) allocRequests(n int) error {
	for _, x := range []**vk.Buffer{&b.req, &b.in, &b.out} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
	var err error
	if b.req, err = b.d.Host(n*4, vk.UsageStorage); err != nil {
		return err
	}
	if b.in, err = b.d.Host(n*b.inStride*4, vk.UsageStorage); err != nil {
		return err
	}
	if b.out, err = b.d.Readback(n*b.outStride*4, vk.UsageStorage); err != nil {
		return err
	}
	b.reqCap = n
	return b.rebuildSet()
}

func (b *Batch) rebuildSet() error {
	if b.set != nil {
		b.set.Close()
		b.set = nil
	}
	var err error
	b.set, err = b.pipe.NewSet([]*vk.Buffer{b.arena, b.req, b.in, b.out})
	return err
}

// Begin opens a round of n requests. It grows the request buffers by
// doubling when n does not fit. Nothing may be in flight.
func (b *Batch) Begin(n int) {
	b.round++
	b.n = n
	b.lastBorn = len(b.births)
	b.births = b.births[:0]
	b.shrinkStaging()
	b.cpuOut = make([][]float32, n)
	if n > b.reqCap {
		if err := b.allocRequests(max(n, 2*b.reqCap)); err != nil {
			panic("gpubrain: growing the request buffers: " + err.Error())
		}
	}
}

// shrinkStaging gives the staging buffer back at its starting size once a
// burst of births is over. It is not in flight at Begin.
func (b *Batch) shrinkStaging() {
	if b.stageCap == stageStart || b.stageCap <= 4*b.lastBorn {
		return
	}
	st, err := b.d.Host(stageStart*slotBytes, vk.UsageTransferSrc)
	if err != nil {
		return // keep the big one
	}
	b.staging.Close()
	b.staging = st
	b.stageCap = stageStart
}

// Set asks for network net to think about in as request i. It is safe to
// call from several goroutines for distinct i. A network that is not on the
// card yet is packed into the staging buffer and given a slot; one that does
// not fit a slot is run on the CPU right away.
func (b *Batch) Set(i int, net *neat.Network, in []float64) {
	if len(in) > b.inStride {
		panic(fmt.Sprintf("gpubrain: %d inputs for a batch built for %d", len(in), b.inStride))
	}
	slot := uint32(net.Tag)
	if net.Tag < 0 {
		slot = b.admit(net)
	}
	if slot == noSlot {
		in32 := make([]float32, len(in))
		for k, x := range in {
			in32[k] = float32(x)
		}
		b.cpuOut[i] = append([]float32(nil), net.Activate(in32)...)
	} else {
		// Missing inputs read as 0, as on the CPU, so the stale values of
		// the previous round must not stay.
		dst := b.in.Floats()[i*b.inStride : (i+1)*b.inStride]
		for k := range dst {
			dst[k] = 0
			if k < len(in) {
				dst[k] = float32(in[k])
			}
		}
	}
	b.req.Uints()[i] = slot
}

// admit gives a newborn a slot, or returns noSlot when the network is too
// big for one. Everything that touches shared state happens under the mutex.
func (b *Batch) admit(net *neat.Network) uint32 {
	f := net.Flat() // reads net only, so outside the mutex
	big := !fits(f)
	b.mu.Lock()
	defer b.mu.Unlock()
	if big {
		b.onCPU++
		return noSlot
	}
	if len(b.free) == 0 {
		b.growArena()
	}
	slot := b.free[len(b.free)-1]
	b.free = b.free[:len(b.free)-1]
	staged := len(b.births)
	if staged >= b.stageCap {
		b.growStaging()
	}
	pack(b.staging.Uints()[staged*SlotWords:(staged+1)*SlotWords], f)
	b.births = append(b.births, birth{slot, int32(staged)})
	net.Tag = slot
	b.owner[slot] = net
	b.uploaded++
	return uint32(slot)
}

// growArena doubles the arena. It must not record on the device: another
// Batch may have a dispatch in flight on the same one. So the old arena is
// kept and Start records the copy into the new one. Growing twice in a round
// drops the intermediate arena, which never held anything.
func (b *Batch) growArena() {
	oldCap := b.capacity
	arena, err := b.d.Local(2*oldCap*slotBytes, usageArena)
	if err != nil {
		panic("gpubrain: growing the arena: " + err.Error())
	}
	if b.prev == nil {
		b.prev, b.prevSize = b.arena, oldCap*slotBytes
	} else {
		b.arena.Close()
	}
	b.arena = arena
	if err := b.rebuildSet(); err != nil {
		panic("gpubrain: rebuilding the set: " + err.Error())
	}
	b.addSlots(oldCap, 2*oldCap)
}

// growStaging doubles the staging buffer and keeps the newborns already in it.
func (b *Batch) growStaging() {
	st, err := b.d.Host(2*b.stageCap*slotBytes, vk.UsageTransferSrc)
	if err != nil {
		panic("gpubrain: growing the staging buffer: " + err.Error())
	}
	copy(st.Bytes(), b.staging.Bytes())
	b.staging.Close()
	b.staging = st
	b.stageCap *= 2
}

// Start frees the slots nobody asked for in ttl rounds, uploads the newborns
// and dispatches one workgroup per request. It returns at once. Slots are
// freed here, before this round's births are copied, and a freed slot is only
// reusable from the next round, so it can never be handed to a birth that is
// already staged.
func (b *Batch) Start() error {
	for _, slot := range b.req.Uints()[:b.n] {
		if slot != noSlot {
			b.lastSeen[slot] = b.round
		}
	}
	b.mu.Lock()
	for s, net := range b.owner {
		if net != nil && b.round-b.lastSeen[s] > b.ttl {
			net.Tag = -1
			b.owner[s] = nil
			b.free = append(b.free, int32(s))
			b.freed++
		}
	}
	b.mu.Unlock()
	push := [3]uint32{uint32(b.n), uint32(b.inStride), uint32(b.outStride)}
	return b.d.Start(func(r *vk.Recorder) {
		if b.prev != nil {
			r.Copy(b.arena, 0, b.prev, b.prevSize)
			r.Barrier()
		}
		for _, bi := range b.births {
			r.CopyFrom(b.arena, int(bi.slot)*slotBytes, b.staging, int(bi.staged)*slotBytes, slotBytes)
		}
		if len(b.births) > 0 {
			r.Barrier()
		}
		if b.n > 0 {
			r.Dispatch(b.set, uint32(b.n), unsafe.Pointer(&push))
		}
	})
}

// Finish waits for the dispatch Start began.
func (b *Batch) Finish() error {
	err := b.d.Wait()
	if b.prev != nil { // its copy has run
		b.prev.Close()
		b.prev = nil
	}
	return err
}

// Out returns the outputs of request i, valid after Finish. The caller reads
// only as many as its network has. The slice is valid until the next Begin.
func (b *Batch) Out(i int) []float32 {
	if b.cpuOut[i] != nil {
		return b.cpuOut[i]
	}
	return b.out.Floats()[i*b.outStride : (i+1)*b.outStride]
}

// Close releases everything the batch holds on the device.
func (b *Batch) Close() {
	if b.set != nil {
		b.set.Close()
	}
	for _, x := range []*vk.Buffer{b.prev, b.arena, b.req, b.in, b.out, b.staging} {
		if x != nil {
			x.Close()
		}
	}
	if b.pipe != nil {
		b.pipe.Close()
	}
}

// Stats returns the networks uploaded, the slots freed and the networks run
// on the CPU since the last call.
func (b *Batch) Stats() (uploaded, freed, cpu int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	uploaded, freed, cpu = b.uploaded, b.freed, b.onCPU
	b.uploaded, b.freed, b.onCPU = 0, 0, 0
	return
}
