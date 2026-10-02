package prep

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
)

const (
	magic   = "NTXT"
	version = 1
)

// Write saves the data in the little endian prep file format.
func (d *Data) Write(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	w.WriteString(magic)
	for _, v := range []any{
		uint32(version), uint32(d.Dim), uint32(d.Vocab()),
		d.QwenID, d.Bytes, d.E,
		uint32(len(d.Train)), d.Train,
		uint32(len(d.Val)), d.Val,
	} {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads a prep file and refuses anything that is not a well-formed one.
func Load(path string) (*Data, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)

	head := make([]byte, len(magic))
	if _, err := io.ReadFull(r, head); err != nil || string(head) != magic {
		return nil, fmt.Errorf("prep: %s is not a prep file", path)
	}
	var ver, dim, vocab uint32
	for _, p := range []*uint32{&ver, &dim, &vocab} {
		if err := binary.Read(r, binary.LittleEndian, p); err != nil {
			return nil, fmt.Errorf("prep: %s: %w", path, err)
		}
	}
	if ver != version {
		return nil, fmt.Errorf("prep: %s has version %d, want %d", path, ver, version)
	}
	// The sizes come from the file, so bound them before allocating.
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if uint64(vocab)*(8+4*uint64(dim)) > uint64(st.Size()) {
		return nil, fmt.Errorf("prep: %s is truncated", path)
	}

	d := &Data{
		Dim:    int(dim),
		QwenID: make([]int32, vocab),
		Bytes:  make([]int32, vocab),
		E:      make([]float32, uint64(vocab)*uint64(dim)),
	}
	for _, v := range []any{d.QwenID, d.Bytes, d.E} {
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("prep: %s: %w", path, err)
		}
	}
	if d.Train, err = readIDs(r, vocab, st.Size()); err != nil {
		return nil, fmt.Errorf("prep: %s: %w", path, err)
	}
	if d.Val, err = readIDs(r, vocab, st.Size()); err != nil {
		return nil, fmt.Errorf("prep: %s: %w", path, err)
	}
	if _, err := r.ReadByte(); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("prep: %s has trailing bytes", path)
	}
	return d, nil
}

func readIDs(r io.Reader, vocab uint32, size int64) ([]int32, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if int64(n)*4 > size {
		return nil, errors.New("truncated")
	}
	ids := make([]int32, n)
	if err := binary.Read(r, binary.LittleEndian, ids); err != nil {
		return nil, err
	}
	for _, id := range ids {
		if id < 0 || uint32(id) >= vocab {
			return nil, fmt.Errorf("token id %d outside the vocabulary of %d", id, vocab)
		}
	}
	return ids, nil
}

// Checksum identifies the embedding a champion was trained with: FNV-1a 64
// over Dim, QwenID and the bits of E.
func Checksum(d *Data) uint64 {
	h := fnv.New64a()
	var b [4]byte
	put := func(v uint32) {
		binary.LittleEndian.PutUint32(b[:], v)
		h.Write(b[:])
	}
	put(uint32(d.Dim))
	for _, q := range d.QwenID {
		put(uint32(q))
	}
	for _, e := range d.E {
		put(math.Float32bits(e))
	}
	return h.Sum64()
}
