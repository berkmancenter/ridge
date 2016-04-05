// Package ridge implements ridge regression.
package ridge

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"

	"math"
)

const BasicallyZero = 1.0e-15

type RidgeRegression struct {
	X            *mat64.Dense
	XSVD         *mat64.SVD
	Y            *mat64.Vector
	L2Penalty    float64
	Coefficients []float64
	Fitted       []float64
	Residuals    []float64
}

// New returns a new ridge regression.
func New(x *mat64.Dense, y *mat64.Vector, l2Penalty float64) *RidgeRegression {
	return &RidgeRegression{
		X:         x,
		Y:         y,
		L2Penalty: l2Penalty,
		Fitted:    make([]float64, y.Len()),
		Residuals: make([]float64, y.Len()),
	}
}

// Regress runs the ridge regression to calculate coefficients.
func (r *RidgeRegression) Regress() {
	r.solveSVD()
	xr, _ := r.X.Dims()

	fitted := mat64.NewVector(xr, nil)
	fitted.MulVec(r.X, mat64.NewVector(len(r.Coefficients), r.Coefficients))
	r.Fitted = fitted.RawVector().Data

	for i := range r.Residuals {
		r.Residuals[i] = r.Y.At(i, 0) - r.Fitted[i]
	}
}

func (r *RidgeRegression) solveSVD() {
	if r.XSVD == nil || r.XSVD.Kind() == 0 {
		r.XSVD = new(mat64.SVD)
		r.XSVD.Factorize(r.X, matrix.SVDThin)
	}

	xr, xc := r.X.Dims()
	xMinDim := int(math.Min(float64(xr), float64(xc)))

	u := mat64.NewDense(xr, xMinDim, nil)
	u.UFromSVD(r.XSVD)

	s := r.XSVD.Values(nil)
	for i := 0; i < len(s); i++ {
		if s[i] < BasicallyZero {
			s[i] = 0
		} else {
			s[i] = s[i] / (s[i]*s[i] + r.L2Penalty)
		}
	}
	d := mat64.NewDense(len(s), len(s), nil)
	setDiag(d, s)

	v := mat64.NewDense(xc, xMinDim, nil)
	v.VFromSVD(r.XSVD)

	uty := mat64.NewVector(xMinDim, nil)
	uty.MulVec(u.T(), r.Y)

	duty := mat64.NewVector(len(s), nil)
	duty.MulVec(d, uty)

	coef := mat64.NewVector(xc, nil)
	coef.MulVec(v, duty)

	r.Coefficients = coef.RawVector().Data
}

func getDiag(mat mat64.Matrix) []float64 {
	r, _ := mat.Dims()
	diag := make([]float64, r)
	for i := range diag {
		diag[i] = mat.At(i, i)
	}
	return diag
}

func setDiag(mat mat64.Mutable, d []float64) {
	for i, v := range d {
		mat.Set(i, i, v)
	}
}
