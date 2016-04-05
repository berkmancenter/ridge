package ridge

import (
	"github.com/gonum/matrix/mat64"

	"fmt"
	"testing"
)

func TestRegress(t *testing.T) {
	for _, test := range testCases {
		r := New(test.x, test.y, test.l2Penalty)
		r.Regress()
		observed := mat64.NewVector(len(r.Coefficients), r.Coefficients)
		if !mat64.EqualApprox(observed, test.expected, 0.001) {
			t.Errorf("Failed %v. Expected\n%v\nbut got\n%v\n",
				test.name, fmtMat(test.expected), fmtMat(observed))
		}
	}
}

func BenchmarkRegress(b *testing.B) {
	test := testCases[1]
	for i := 0; i < b.N; i++ {
		r := New(test.x, test.y, test.l2Penalty)
		b.ResetTimer()
		r.Regress()
	}
}

func fmtMat(mat mat64.Matrix) fmt.Formatter {
	return mat64.Formatted(mat, mat64.Excerpt(2), mat64.Squeeze())
}
