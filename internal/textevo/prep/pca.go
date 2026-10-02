package prep

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// pca centres the rows, projects them on the dim directions of largest
// variance and scales each direction to unit variance, so that the network's
// inputs are all of the same size whatever the embedding's own scale.
func pca(x []float64, n, width, dim int) []float32 {
	for j := range width {
		var m float64
		for i := range n {
			m += x[i*width+j]
		}
		m /= float64(n)
		for i := range n {
			x[i*width+j] -= m
		}
	}

	xc := mat.NewDense(n, width, x)
	cov := mat.NewSymDense(width, nil)
	cov.SymOuterK(1/float64(n-1), xc.T())
	var es mat.EigenSym
	if !es.Factorize(cov, true) {
		panic("prep: eigen decomposition of the covariance failed")
	}
	var vec mat.Dense
	es.VectorsTo(&vec)

	// Eigenvalues ascend, so the wanted directions are the last columns, the
	// largest first.
	v := mat.NewDense(width, dim, nil)
	for j := range dim {
		v.SetCol(j, mat.Col(nil, width-1-j, &vec))
	}
	var p mat.Dense
	p.Mul(xc, v)

	out := make([]float32, n*dim)
	for j := range dim {
		var s float64
		for i := range n {
			s += p.At(i, j) * p.At(i, j)
		}
		std := math.Sqrt(s / float64(n-1))
		if std < 1e-12 {
			std = 1
		}
		for i := range n {
			out[i*dim+j] = float32(p.At(i, j) / std)
		}
	}
	return out
}
