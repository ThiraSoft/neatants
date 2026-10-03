// Text evolution: NEAT networks that read Qwen embeddings and predict the next
// token of TinyShakespeare, scored in bits per byte.
//
//	make text
//	./neattext -prep                 # tokenize the corpus once
//	./neattext -baselines            # n-gram reference scores
//	./neattext -pop 200 -gens 12     # evolve, on the GPU
//	./neattext -state 4              # evolve networks that read four state banks
//	./neattext -sample runs/<dir>/champion.json
//
// -data picks the prepared file in every mode, so several dimensions can sit
// side by side: ./neattext -prep -dim 32 -data data/prep32.bin.
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

const corpusPath = "data/tinyshakespeare.txt"

var (
	prepMode  = flag.Bool("prep", false, "tokenize data/tinyshakespeare.txt and write the -data file")
	dataPath  = flag.String("data", "data/prep.bin", "prepared corpus: written by -prep, read by every other mode")
	baselines = flag.Bool("baselines", false, "print the n-gram baselines in bits per byte")
	sampleOf  = flag.String("sample", "", "champion JSON to write text with")
	esOf      = flag.String("es", "", "softmax champion JSON whose link weights OpenAI-ES tunes, its topology fixed (-pop perturbations a step)")
	esSigma   = flag.Float64("es-sigma", 0.05, "-es: standard deviation of the weight perturbations")
	esLR      = flag.Float64("es-lr", 0.01, "-es: Adam step size")
	gguf      = flag.String("gguf", "/mnt/data/LLMs_models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf", "Qwen model: tokenizer and embeddings")
	dim       = flag.Int("dim", 128, "-prep only: embedding dimensions kept by the PCA (multiple of 32); the other modes use the file's")
	pop       = flag.Int("pop", 1000, "population size")
	windows   = flag.Int("windows", 4, "windows per genome per generation")
	length    = flag.Int("len", 128, "tokens per window")
	warm      = flag.Int("warm", model.Warm, "warm-up ticks per window, not scored")
	gens      = flag.Int("gens", 0, "generations to run, 0 for no limit")
	hebb      = flag.Float64("hebb", neat.HebbRate, "rate at which a mutation makes a link plastic")
	links     = flag.Int("links", neat.MinimalLinks, "inputs each output reads in a first-generation genome (the dim for a dense map)")
	wmut      = flag.Float64("wmut", neat.WeightsPerMutation, "weights a mutation perturbs on average")
	score     = flag.String("score", "softmax", "how a prediction is scored: softmax over the vocabulary, or tree, one binary decision a level of a tree of the tokens")
	input     = flag.String("input", "emb", "-score tree only: what a network reads of a token, emb (its embedding), code (its path in the tree) or both")
	banks     = flag.Int("state", 0, "state banks every network reads, 0 to 4: running averages of the embeddings at four time scales, written through gates")
	learnOn   = flag.Bool("learn", false, "Lamarckian learning: every genome takes a step of backpropagation through time on its windows each generation, and its children inherit the learned weights")
	lrW       = flag.Float64("lr", 0.05, "-learn: step of the link weights (normalized gradient descent)")
	lrEta     = flag.Float64("lr-eta", 0.002, "-learn: step of the plasticity rates")
	lrTrait   = flag.Float64("lr-trait", 0.01, "-learn: step of the logit scale trait")
	widen     = flag.Float64("widen", 0, "chance per mutation to add a block of hidden neurons that read the inputs, with outgoing weights at zero (for -learn)")
	widenN    = flag.Int("widen-nodes", neat.WidenNodes, "-widen: neurons a block")
	widenIn   = flag.Int("widen-in", neat.WidenIn, "-widen: inputs each neuron of a block reads")
	widenOut  = flag.Int("widen-out", neat.WidenOut, "-widen: neurons that read each neuron of a block")
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
	Banks    int          `json:"banks"`
	Score    string       `json:"score,omitempty"` // "tree", or empty for softmax
	Input    string       `json:"input,omitempty"` // under tree: "code" or "both", or empty for the embedding
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
	case *esOf != "":
		runES()
	default:
		runEvolve()
	}
}

func load() *prep.Data {
	d, err := prep.Load(*dataPath)
	if err != nil {
		fail(1, "%v (run ./neattext -prep -data %s first)", err, *dataPath)
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
	if err := d.Write(*dataPath); err != nil {
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
		fail(1, "%s was evolved on another corpus preparation (dim %d, checksum %016x; %s has dim %d, checksum %016x)",
			*sampleOf, c.Dim, c.Checksum, *dataPath, d.Dim, prep.Checksum(d))
	}
	// The genome's inputs say how many banks it reads; a champion that
	// disagrees with its own record was not written by this program.
	in, out := c.Genome.NumInputs, c.Genome.NumOutputs
	var tr *model.Tree
	rows := d
	switch c.Score {
	case "":
	case "tree":
		tr = model.BuildTree(d)
		switch c.Input {
		case "":
		case model.InputCode, model.InputBoth:
			rows = model.InputData(d, tr, c.Input)
		default:
			fail(1, "%s: unknown input %q", *sampleOf, c.Input)
		}
	default:
		fail(1, "%s: unknown score %q", *sampleOf, c.Score)
	}
	s := model.Shape{Dim: rows.Dim, Banks: c.Banks}
	if tr != nil && tr.Depth != rows.Dim {
		s.Code = tr.Depth
	}
	if c.Banks < 0 || c.Banks > model.MaxBanks || in != s.Inputs() || out != s.Outputs() {
		fail(1, "%s: a genome of %d inputs and %d outputs is not a network of dim %d with %d state banks", *sampleOf, in, out, c.Dim, c.Banks)
	}
	g, voc := openModel()
	defer g.Close()
	ids, err := encodePrompt(voc, d, *prompt)
	if err != nil {
		fail(1, "%v", err)
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	text := *prompt
	for _, id := range sample(c.Genome, rows, tr, ids, *n, *temp, rng) {
		text += voc.Decode([]int32{d.QwenID[id]}, false)
	}
	fmt.Println(text)
}
