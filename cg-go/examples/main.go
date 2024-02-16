package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/mukappalambda/conjugate-gradient/cg-go/cg"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	n    = flag.Int("n", 10, "number of row (or column) of the positive-definite matrix A")
	seed = flag.Uint64("seed", 42, "seed")
	tol  = flag.Float64("tol", 1e-6, "stopping criterion")
)

func main() {
	flag.Parse()

	src := rand.NewSource(*seed)
	A, err := NewPDDense(*n, src)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(mat.Formatted(A))
	x0 := mat.NewVecDense(*n, nil)
	data := make([]float64, *n)
	dist := distuv.Normal{
		Mu:    5,
		Sigma: 2,
		Src:   src,
	}
	for i := range data {
		data[i] = dist.Rand()
	}
	b := mat.NewVecDense(*n, data)
	fmt.Println("b transpose: ", mat.Formatted(b.T()))

	c := cg.NewContainer(A, x0, b)

	xhat, converged := cg.ConjugateGradient(c, *tol)

	if !converged {
		log.Println("not converged.")
		fmt.Println("xhat transpose: ", mat.Formatted(xhat.T()))
		return
	}

	fmt.Println("xhat transpose: ", mat.Formatted(xhat.T()))
	bhat := mat.NewVecDense(*n, nil)
	bhat.MulVec(A, xhat)
	fmt.Println("A @ xhat = b ?: ", mat.EqualApprox(b, bhat, 1e-6))
}

func NewPDDense(n int, src rand.Source) (*mat.Dense, error) {
	data := make([]float64, n*n)
	dist := distuv.Uniform{
		Min: 0,
		Max: 1,
		Src: src,
	}

	for i := range data {
		data[i] = dist.Rand()
	}

	sigmas := make([]float64, n)

	for i := range sigmas {
		sigmas[i] = 1 + dist.Rand()
	}

	A := mat.NewDense(n, n, data)
	var svd mat.SVD
	ok := svd.Factorize(A, mat.SVDFull)

	if !ok {
		return nil, fmt.Errorf("failed to execute SVD")
	}

	var U mat.Dense
	svd.UTo(&U)
	m := mat.NewDense(n, n, nil)
	m.Mul(&U, mat.NewDiagDense(n, sigmas))
	m.Mul(m, (&U).T())
	return m, nil
}
