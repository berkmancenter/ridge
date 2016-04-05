package ridge

import "github.com/gonum/matrix/mat64"

type testCase struct {
	name      string
	x         *mat64.Dense
	y         *mat64.Vector
	l2Penalty float64
	expected  *mat64.Vector
}

var testCases = []testCase{
	{
		name:      "more samples than features",
		x:         mat64.NewDense(3, 2, []float64{0, 0, 0, 0, 1, 1}),
		y:         mat64.NewVector(3, []float64{0, 0.1, 1}),
		l2Penalty: 1.0,
		expected:  mat64.NewVector(2, []float64{0.27142857, 0.27142857}),
	},
	{
		name:      "more features than samples",
		x:         mat64.NewDense(2, 3, []float64{1, 0, 0, 0, 2, 3}),
		y:         mat64.NewVector(2, []float64{0.1, 1}),
		l2Penalty: 1.0,
		expected:  mat64.NewVector(3, []float64{-0.05625, 0.1125, 0.16875}),
	},
	{
		name:      "non-default l2 penalty",
		x:         mat64.NewDense(2, 3, []float64{1, 0, 0, 0, 2, 3}),
		y:         mat64.NewVector(2, []float64{0.1, 1}),
		l2Penalty: 0.5,
		expected:  mat64.NewVector(3, []float64{-0.06, 0.12, 0.18}),
	},
	{
		name:      "same as OLS",
		x:         mat64.NewDense(3, 2, []float64{0, 0, 0, 0, 1, 1}),
		y:         mat64.NewVector(3, []float64{0, 0.1, 1}),
		l2Penalty: 0.0,
		expected:  mat64.NewVector(2, []float64{0.95, 0.0}),
	},
}
