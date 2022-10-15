package gograd_nd

import "fmt"

// ----------------------------------- Notes: -----------------------------------

// Need to refactor code to use tensornd structs instead of tensor structs.

//  ----------------------------------- Data Structures:  -----------------------------------

// Comp Graph Node
type node struct {
	left         *node
	right        *node
	op           string
	tensor       *tensor
	grad         *tensor
	l2           int
	l            int
	require_grad bool
}

// Will be implementing an alternative tensor design which uses a single slice instead of slices of pointers of slices of pointers of...
type tensor struct {
	data   []float64 // all tensor data in a single slice
	dimlen []int     // [i] = length of dimension i
	diminc []int     // [i] = increment of dimension i
}

//  ----------------------------------- Auxiliary Functions:  -----------------------------------

// NOTE: not currently checking for errors in these functions!!!

func (t *tensor) get(ind []int) float64 {
	index := 0
	for i, x := range ind {
		index += t.diminc[i] * x
	}
	return t.data[index]
}

func (t *tensor) shape() []int {
	return t.dimlen
}

func (t *tensor) transpose(dim1 int, dim2 int) {
	t.diminc[dim1], t.diminc[dim2] = t.diminc[dim2], t.diminc[dim1]
	t.dimlen[dim1], t.dimlen[dim2] = t.dimlen[dim2], t.dimlen[dim1]
}

// ----------------------------------- TESTING (will be removed when done) -----------------------------------

func Test() {
	data := []float64{1, 2, 3, 4}
	dl := []int{4, 2}
	di := []int{2, 1}
	a := tensor{data, dl, di}
	g := []int{0, 0}
	gg := []int{0, 1}
	fmt.Println(a.get(g), a.get(gg))
	g = []int{1, 0}
	gg = []int{1, 1}
	fmt.Println(a.get(g), a.get(gg))
	fmt.Println("transpose")
	a.transpose(0, 1)
	g = []int{0, 0}
	gg = []int{0, 1}
	fmt.Println(a.get(g), a.get(gg))
	g = []int{1, 0}
	gg = []int{1, 1}
	fmt.Println(a.get(g), a.get(gg))

}
