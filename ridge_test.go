package ridge

import (
	"gonum.org/v1/gonum/mat"

	"testing"
)

func TestRegress(t *testing.T) {
	for _, test := range testCases {
		if test.skip {
			continue
		}
		r := New(test.data.x, test.data.y, test.l2Penalty)
		r.Regress()
		observed := r.Coefficients
		if !mat.EqualApprox(observed, test.expected, 0.001) {
			t.Errorf("Failed %v. Expected\n%v\nbut got\n%v\n",
				test.name, fmtMat(test.expected), fmtMat(observed))
		}
	}
}

func BenchmarkRegress(b *testing.B) {
	test := testCases[4]
	for i := 0; i < b.N; i++ {
		r := New(test.data.x, test.data.y, test.l2Penalty)
		r.Regress()
	}
}
