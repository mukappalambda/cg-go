package cg

import (
	"gonum.org/v1/gonum/mat"
)

type Container struct {
	A            *mat.Dense
	x            *mat.VecDense
	b            *mat.VecDense
	Grad         *mat.VecDense
	Dir          *mat.VecDense
	isInitialDir bool
}

func NewContainer(A *mat.Dense, x *mat.VecDense, b *mat.VecDense) *Container {
	c := &Container{
		A:            A,
		x:            x,
		b:            b,
		Grad:         b,
		Dir:          b,
		isInitialDir: true,
	}
	c.Grad = c.gradient()
	dir := mat.NewVecDense(c.Grad.Len(), nil)
	dir.ScaleVec(-1.0, c.Grad)
	c.Dir = dir
	return c
}

func ConjugateGradient(c *Container, tol float64) (sol *mat.VecDense, converged bool) {
	maxiters, _ := c.A.Dims()

	if c.Converged(tol) {
		return c.x, true
	}

	for i := 0; i <= maxiters; i++ {
		c.UpdateDirection()
		c.UpdateSolution()
		c.UpdateGradient()

		if c.Converged(tol) {
			return c.x, true
		}
	}
	return c.x, false
}

func (c *Container) Alpha() float64 {
	eye := Eye(c.Grad.Len())
	num := mat.Inner(c.Grad, eye, c.Dir)
	denom := mat.Inner(c.Dir, c.A, c.Dir)
	return -num / denom
}

func Eye(n int) *mat.Dense {
	eye := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		eye.Set(i, i, 1.0)
	}
	return eye
}

func (c *Container) Beta() float64 {
	num := mat.Inner(c.Grad, c.A, c.Dir)
	denom := mat.Inner(c.Dir, c.A, c.Dir)
	return num / denom
}

func (c *Container) UpdateDirection() {
	if c.isInitialDir {
		c.isInitialDir = false
		return
	}
	negGrad := mat.NewVecDense(c.Grad.Len(), nil)
	negGrad.ScaleVec(-1.0, c.Grad)
	c.Dir.AddScaledVec(negGrad, c.Beta(), c.Dir)
}

func (c *Container) UpdateSolution() {
	c.x.AddScaledVec(c.x, c.Alpha(), c.Dir)
}

func (c *Container) UpdateGradient() {
	c.Grad = c.gradient()
}

func (c *Container) gradient() *mat.VecDense {
	grad := mat.NewVecDense(c.Grad.Len(), nil)
	grad.MulVec(c.A, c.x)
	grad.AddScaledVec(grad, -1.0, c.b)
	return grad
}

func (c *Container) Converged(tol float64) bool {
	return c.Grad.Norm(2) < tol
}
