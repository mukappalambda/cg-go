package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/mukappalambda/conjugate-gradient/cg-go/cg"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	n   = flag.Int("n", 10, "number of row (or column) of the positive-definite matrix A")
	tol = flag.Float64("tol", 1e-6, "stopping criterion")
)

func main() {
	flag.Parse()

	A, err := NewPDDense(*n)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(mat.Formatted(A))
	x0 := mat.NewVecDense(*n, nil)
	data := make([]float64, *n)
	dist := distuv.Normal{
		Mu:    5,
		Sigma: 2,
	}
	for i := range data {
		data[i] = dist.Rand()
	}
	b := mat.NewVecDense(*n, data)
	fmt.Println("b", mat.Formatted(b))

	c := cg.NewContainer(A, x0, b)

	sol, converged := cg.ConjugateGradient(c, *tol)

	if !converged {
		log.Println("not converged.")
		fmt.Println(mat.Formatted(sol))
		return
	}

	fmt.Println("sol", mat.Formatted(sol.T()))
}

func NewPDDense(n int) (*mat.Dense, error) {
	data := make([]float64, n*n)
	dist := distuv.Uniform{
		Min: 0,
		Max: 1,
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
