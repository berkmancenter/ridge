// Package ridge implements ridge regression.
package ridge

import (
	"gonum.org/v1/gonum/mat"

	"fmt"
	"math"
)

const BasicallyZero = 1.0e-15

func fmtMat(m mat.Matrix) fmt.Formatter {
	return mat.Formatted(m, mat.Excerpt(2), mat.Squeeze())
}

type RidgeRegression struct {
	X            *mat.Dense
	XSVD         *mat.SVD
	U            *mat.Dense
	D            *mat.Dense
	V            *mat.Dense
	XScaled      *mat.Dense
	Y            *mat.VecDense
	Scales       *mat.VecDense
	L2Penalty    float64
	Coefficients *mat.VecDense
	Fitted       []float64
	Residuals    []float64
	StdErrs      []float64
}

// New returns a new ridge regression.
func New(x *mat.Dense, y *mat.VecDense, l2Penalty float64) *RidgeRegression {
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
	r.scaleX()
	r.solveSVD()
	xr, _ := r.X.Dims()

	fitted := mat.NewVecDense(xr, nil)
	fitted.MulVec(r.X, r.Coefficients)
	r.Fitted = fitted.RawVector().Data

	for i := range r.Residuals {
		r.Residuals[i] = r.Y.At(i, 0) - r.Fitted[i]
	}

	r.calcStdErr()
}

func (r *RidgeRegression) scaleX() {
	xr, xc := r.X.Dims()
	scaleData := make([]float64, xr)
	scalar := 1.0 / float64(xr)
	for i := range scaleData {
		scaleData[i] = scalar
	}
	scaleMat := mat.NewDense(1, xr, scaleData)
	sqX := mat.NewDense(xr, xc, nil)
	sqX.MulElem(r.X, r.X)

	scales := mat.NewDense(1, xc, nil)
	scales.Mul(scaleMat, sqX)
	sqrtElem := func(i, j int, v float64) float64 { return math.Sqrt(v) }
	scales.Apply(sqrtElem, scales)
	r.Scales = mat.NewVecDense(xc, scales.RawRowView(0))
	r.XScaled = mat.NewDense(xr, xc, nil)
	scale := func(i, j int, v float64) float64 { return v / r.Scales.At(j, 0) }
	r.XScaled.Apply(scale, r.X)
}

func (r *RidgeRegression) solveSVD() {
	if r.XSVD == nil || r.XSVD.Kind() == 0 {
		r.XSVD = new(mat.SVD)
		r.XSVD.Factorize(r.XScaled, mat.SVDThin)
	}

	xr, xc := r.XScaled.Dims()
	xMinDim := int(math.Min(float64(xr), float64(xc)))

	u := mat.NewDense(xr, xMinDim, nil)
	r.XSVD.UTo(u)
	r.U = u

	s := r.XSVD.Values(nil)
	for i := 0; i < len(s); i++ {
		if s[i] < BasicallyZero {
			s[i] = 0
		} else {
			s[i] = s[i] / (s[i]*s[i] + r.L2Penalty)
		}
	}
	d := mat.NewDense(len(s), len(s), nil)
	setDiag(d, s)
	r.D = d

	v := mat.NewDense(xc, xMinDim, nil)
	r.XSVD.VTo(v)
	r.V = v

	uty := mat.NewVecDense(xMinDim, nil)
	uty.MulVec(u.T(), r.Y)

	duty := mat.NewVecDense(len(s), nil)
	duty.MulVec(d, uty)

	coef := mat.NewVecDense(xc, nil)
	coef.MulVec(v, duty)

	r.Coefficients = mat.NewVecDense(xc, nil)
	r.Coefficients.DivElemVec(coef, r.Scales)
}

// http://stats.stackexchange.com/a/2126
// http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
func (r *RidgeRegression) calcStdErr() {
	xr, xc := r.X.Dims()
	xMinDim := int(math.Min(float64(xr), float64(xc)))
	errVari := 0.0
	for _, v := range r.Residuals {
		errVari += v * v
	}
	errVari /= float64(xr - xc)
	errVariMat := mat.NewDense(xr, xr, nil)
	for i := 0; i < xr; i++ {
		errVariMat.Set(i, i, errVari)
	}

	//    V                   D                   UT        Z
	// xc x xMinDim * xMinDim x xMinDim * xMinDim x xr = xc x xr
	vd := mat.NewDense(xc, xMinDim, nil)
	vd.Mul(r.V, r.D)
	z := mat.NewDense(xc, xr, nil)
	z.Mul(vd, r.U.T())

	//    Z       ErrVar      ZT
	// xc x xr * xr x xr * xr x xc
	zerr := mat.NewDense(xc, xr, nil)
	zerr.Mul(z, errVariMat)
	coefCovarMat := mat.NewDense(xc, xc, nil)
	coefCovarMat.Mul(zerr, z.T())
	r.StdErrs = getDiag(coefCovarMat)
}

func getDiag(mat mat.Matrix) []float64 {
	r, _ := mat.Dims()
	diag := make([]float64, r)
	for i := range diag {
		diag[i] = mat.At(i, i)
	}
	return diag
}

func setDiag(mat mat.Mutable, d []float64) {
	for i, v := range d {
		mat.Set(i, i, v)
	}
}
