package baseline

import (
	"math"
	"testing"

	"github.com/ThiraSoft/neatants/internal/textevo/model"
)

func TestOrdersImproveOnRepetitiveText(t *testing.T) {
	d := model.Synthetic(20, 8, 10, 1)
	// A periodic text: a trigram model predicts it almost perfectly.
	d.Train = make([]int32, 5000)
	for i := range d.Train {
		d.Train[i] = int32(i % 7)
	}
	d.Val = d.Train[:700]
	r := Evaluate(d)
	if !(r.Unigram > r.Bigram && r.Bigram >= r.Trigram && r.Trigram < 0.3) {
		t.Fatalf("%+v", r)
	}
	if math.IsNaN(r.Unigram) || math.IsInf(r.Trigram, 0) {
		t.Fatalf("%+v", r)
	}
}
