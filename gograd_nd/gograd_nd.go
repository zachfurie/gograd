package gograd_nd

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
	l      int       // = len(data)
	dims   int       // = len(dimlen)  = len(diminc)
}

//  ----------------------------------- Auxiliary Functions:  -----------------------------------
