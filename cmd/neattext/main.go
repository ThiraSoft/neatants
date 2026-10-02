// Text evolution: NEAT networks that read Qwen embeddings and predict the next
// token of TinyShakespeare, scored in bits per byte.
//
//	make text
//	./neattext -prep                 # tokenize the corpus once
//	./neattext -baselines            # n-gram reference scores
//	./neattext -pop 200 -gens 12     # evolve, on the GPU
//	./neattext -sample runs/<dir>/champion.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/neatants/internal/textevo/baseline"
	"github.com/ThiraSoft/neatants/internal/textevo/model"
	"github.com/ThiraSoft/neatants/internal/textevo/prep"
	"github.com/ThiraSoft/neatants/neat"
)

const (
	corpusPath = "data/tinyshakespeare.txt"
	prepPath   = "data/prep.bin"
)

var (
	prepMode  = flag.Bool("prep", false, "tokenize data/tinyshakespeare.txt and write data/prep.bin")
	baselines = flag.Bool("baselines", false, "print the n-gram baselines in bits per byte")
	sampleOf  = flag.String("sample", "", "champion JSON to write text with")
	gguf      = flag.String("gguf", "/mnt/data/LLMs_models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf", "Qwen model: tokenizer and embeddings")
	dim       = flag.Int("dim", 128, "embedding dimensions kept by the PCA (multiple of 32)")
	pop       = flag.Int("pop", 1000, "population size")
	windows   = flag.Int("windows", 4, "windows per genome per generation")
	length    = flag.Int("len", 128, "tokens per window")
	warm      = flag.Int("warm", model.Warm, "warm-up ticks per window, not scored")
	gens      = flag.Int("gens", 0, "generations to run, 0 for no limit")
	hebb      = flag.Float64("hebb", neat.HebbRate, "rate at which a mutation makes a link plastic")
	cpu       = flag.Bool("cpu", false, "evaluate on the CPU reference instead of the GPU")
	out       = flag.String("out", "", "run directory (default runs/<date-time>)")
	n         = flag.Int("n", 200, "tokens to sample")
	temp      = flag.Float64("temp", 1, "sampling temperature")
	prompt    = flag.String("prompt", "ROMEO:", "sampling prompt")
)

// champion is what a run saves: the genome and what ties it to its corpus.
type champion struct {
	Genome   *neat.Genome `json:"genome"`
	Dim      int          `json:"dim"`
	Checksum uint64       `json:"checksum"`
	ValBPB   float64      `json:"val_bpb"`
	Gen      int          `json:"gen"`
}

func fail(code int, format string, a ...any) {
	fmt.Fprintf(os.Stderr, "neattext: "+format+"\n", a...)
	os.Exit(code)
}

func main() {
	flag.Parse()
	switch {
	case *prepMode:
		runPrep()
	case *baselines:
		d := load()
		printBaselines(d)
	case *sampleOf != "":
		runSample()
	default:
		runEvolve()
	}
}

func load() *prep.Data {
	d, err := prep.Load(prepPath)
	if err != nil {
		fail(1, "%v (run ./neattext -prep first)", err)
	}
	return d
}

func printBaselines(d *prep.Data) {
	b := baseline.Evaluate(d)
	fmt.Printf("baselines (bits per byte, validation): unigram %.4f  bigram %.4f  trigram %.4f\n", b.Unigram, b.Bigram, b.Trigram)
}

// openModel opens the GGUF and its tokenizer. The caller closes the file.
func openModel() (*tensors.GGUF, *bytebpe.Vocab) {
	g, err := tensors.OpenGGUF(*gguf)
	if err != nil {
		fail(1, "%v", err)
	}
	voc, err := bytebpe.Load(g)
	if err != nil {
		g.Close()
		fail(1, "%v", err)
	}
	return g, voc
}

func runPrep() {
	if *dim%32 != 0 {
		fail(2, "-dim must be a multiple of 32")
	}
	start := time.Now()
	raw, err := os.ReadFile(corpusPath)
	if err != nil {
		fail(1, "%v (make data)", err)
	}
	train, val := prep.Split(raw)
	g, voc := openModel()
	defer g.Close()
	embd, err := g.Get("token_embd.weight")
	if err != nil {
		fail(1, "%v", err)
	}
	d, err := prep.Build(train, val, voc, embd, *dim)
	if err != nil {
		fail(1, "%v", err)
	}
	if err := d.Write(prepPath); err != nil {
		fail(1, "%v", err)
	}
	fmt.Printf("prep: %d active tokens, dim %d, %d train tokens, %d val tokens, checksum %016x, %s\n",
		d.Vocab(), d.Dim, len(d.Train), len(d.Val), prep.Checksum(d), time.Since(start).Round(time.Millisecond))
}

func runSample() {
	d := load()
	raw, err := os.ReadFile(*sampleOf)
	if err != nil {
		fail(1, "%v", err)
	}
	var c champion
	if err := json.Unmarshal(raw, &c); err != nil {
		fail(1, "%s: %v", *sampleOf, err)
	}
	if c.Genome == nil {
		fail(1, "%s holds no genome", *sampleOf)
	}
	if c.Dim != d.Dim || c.Checksum != prep.Checksum(d) {
		fail(1, "%s was evolved on another corpus preparation (dim %d, checksum %016x; prep.bin has dim %d, checksum %016x)",
			*sampleOf, c.Dim, c.Checksum, d.Dim, prep.Checksum(d))
	}
	g, voc := openModel()
	defer g.Close()
	ids, err := encodePrompt(voc, d, *prompt)
	if err != nil {
		fail(1, "%v", err)
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	text := *prompt
	for _, id := range sample(c.Genome, d, ids, *n, *temp, rng) {
		text += voc.Decode([]int32{d.QwenID[id]}, false)
	}
	fmt.Println(text)
}
