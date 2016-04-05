// Package ridge implements ridge regression.
package ridge

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type RidgeRegression struct {
	X              *mat64.Dense
	XSVD           *mat64.SVD
	Y              *mat64.Vector
	L2penalty      float64
	Coefficients   []float64
	StandardErrors []float64
	Fitted         []float64
	Residuals      []float64
}

// New returns a new ridge regression.
func New(x *mat64.Dense, y *mat64.Vector) *RidgeRegression {
	return &RidgeRegression{
		X:         x,
		Y:         y,
		L2penalty: 0,
		Fitted:    make([]float64, y.Len()),
		Residuals: make([]float64, y.Len()),
	}
}

// Regress runs the ridge regression to calculate coefficients.
func (r *RidgeRegression) Regress(l2penalty float64) {
	if r.XSVD == nil || r.XSVD.Kind() == 0 {
		r.XSVD = new(mat64.SVD)
		r.XSVD.Factorize(r.X, matrix.SVDThin)
	}

	xr, xc := r.X.Dims()
	v := mat64.NewDense(xc, xc, nil)
	v.VFromSVD(r.XSVD)
	s := r.XSVD.Values(nil)
	u := mat64.NewDense(xr, xc, nil)
	u.UFromSVD(r.XSVD)

	for i := 0; i < len(s); i++ {
		s[i] = s[i] / (s[i]*s[i] + l2penalty)
	}
	sMat := mat64.NewDense(len(s), len(s), nil)
	setDiag(sMat, s)

	vs := mat64.NewDense(xc, len(s), nil)
	vs.Mul(v, sMat)
	z := mat64.NewDense(xc, xr, nil)
	z.Mul(vs, u.T())

	coef := mat64.NewVector(xc, nil)
	coef.MulVec(z, r.Y)
	r.Coefficients = coef.RawVector().Data

	fitted := mat64.NewVector(xr, nil)
	fitted.MulVec(r.X, coef)
	r.Fitted = fitted.RawVector().Data

	errorVariance := 0.0
	for i := range r.Residuals {
		r.Residuals[i] = r.Y.At(i, 0) - r.Fitted[i]
		errorVariance += r.Residuals[i] * r.Residuals[i]
	}
	errorVariance /= float64(xr - xc)

	errorVarMat := mat64.NewDense(xr, xr, nil)
	diag := make([]float64, xr)
	for i := range diag {
		diag[i] = errorVariance
	}
	setDiag(errorVarMat, diag)

	zerr := mat64.NewDense(xc, xr, nil)
	zerr.Mul(z, errorVarMat)
	coefCovarMat := mat64.NewDense(xc, xc, nil)
	coefCovarMat.Mul(zerr, z.T())
	r.StandardErrors = getDiag(coefCovarMat)
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
