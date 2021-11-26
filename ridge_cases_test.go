package ridge

import "gonum.org/v1/gonum/mat"

type testCase struct {
	name      string
	skip      bool
	data      dataSet
	l2Penalty float64
	expected  *mat.VecDense
}

type dataSet struct {
	x *mat.Dense
	y *mat.VecDense
}

// These are run against lm.ridge in package MASS in R.
var testCases = []testCase{
	{
		name: "more samples than features",
		data: dataSet{
			x: mat.NewDense(3, 2, []float64{0, 0, 0, 0, 1, 1}),
			y: mat.NewVecDense(3, []float64{0, 0.1, 1}),
		},
		l2Penalty: 1.0,
		expected:  mat.NewVecDense(2, []float64{0.4285714, 0.4285714}),
	},
	{
		name: "more features than samples",
		data: dataSet{
			x: mat.NewDense(2, 3, []float64{1, 0, 0, 0, 2, 3}),
			y: mat.NewVecDense(2, []float64{0.1, 1}),
		},
		l2Penalty: 1.0,
		expected:  mat.NewVecDense(3, []float64{0.06666667, 0.20000000, 0.13333333}),
	},
	{
		name: "non-default l2 penalty",
		data: dataSet{
			x: mat.NewDense(2, 3, []float64{1, 0, 0, 0, 2, 3}),
			y: mat.NewVecDense(2, []float64{0.1, 1}),
		},
		l2Penalty: 0.5,
		expected:  mat.NewVecDense(3, []float64{0.0800000, 0.2222222, 0.1481481}),
	},
	{
		// This isn't the same as OLS in R. Not sure why. I don't try to match OLS,
		// but instead match the lm.ridge behavior.
		name: "same as OLS",
		data: dataSet{
			x: mat.NewDense(2, 3, []float64{1, 0, 0, 0, 2, 3}),
			y: mat.NewVecDense(2, []float64{0.1, 1}),
		},
		l2Penalty: 0.0,
		expected:  mat.NewVecDense(3, []float64{0.1000000, 0.2500000, 0.1666667}),
	},
	{
		name:      "iris data",
		l2Penalty: 1.0,
		data:      iris,
		expected:  mat.NewVecDense(3, []float64{1.1664384, 0.6854012, -0.2890332}),
	},
	{
		name:      "iris data different l2 penalty",
		l2Penalty: 0.5,
		data:      iris,
		expected:  mat.NewVecDense(3, []float64{1.1523371, 0.7729191, -0.5160199}),
	},
}
