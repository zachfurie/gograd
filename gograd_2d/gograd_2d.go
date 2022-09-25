package gograd_2d

import (
	"fmt"
	"math/rand"
)

// ----------------------------------- Notes: -----------------------------------

// If all nodes are passing a tensor to next node during forward pass,
// prob don't need to have weights as inputs to OP funcs.

// Need to refactor code to use tensornd structs instead of tensor structs.

//  ----------------------------------- Data Structures:  -----------------------------------

// Comp Graph Node
type node struct {
	left   *node
	right  *node
	op     string
	tensor *tensor
	grad   *tensor
	l2     int
	l      int
}

// 2d tensor
type tensor struct {
	data []*[]float32
	l2   int
	l    int
	// grad bool
}

// nd tensor (will eventually only use this tensor struct)
type tensornd struct {
	data     []*tensornd
	last_dim *[]float32
	l        int // l is length of either data or last_dim, since tensors will only have either one or the other.
	// for all except last dimension, last_dim is nil
	// see shape() for more info
}

//  ----------------------------------- Auxiliary Functions:  -----------------------------------

// Returns a 2d tensor of size (l2,l)
func zeros(l2 int, l int) tensor {
	zero_grad_pts := make([]*[]float32, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float32, l)
		for j := range zero_grad_data {
			zero_grad_data[j] = 0
		}
		zero_grad_pts[i] = &zero_grad_data
	}
	return tensor{zero_grad_pts, l2, l}
}

// Returns a 2d tensor of size (l2,l)
func ones(l2 int, l int) tensor {
	zero_grad_pts := make([]*[]float32, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float32, l)
		for j := range zero_grad_data {
			zero_grad_data[j] = 1
		}
		zero_grad_pts[i] = &zero_grad_data
	}
	return tensor{zero_grad_pts, l2, l}
}

// Returns shape of tensor t
func shape(t *tensornd) []int {
	root := t
	num_dims := 1
	for root.last_dim == nil {
		num_dims += 1
		root = root.data[0]
	}
	shape_ret := make([]int, num_dims)
	root = t
	i := 0
	for root.last_dim == nil {
		shape_ret[i] = len(root.data)
		root = root.data[0]
	}
	shape_ret[num_dims-1] = len(*root.last_dim)
	return shape_ret
}

// Creates a new tensor that is a copy of the input tensor with the input dimensions transposed.
func transpose(t *tensornd, dim1 int, dim2 int) *tensornd {
	// check shape to make sure dims are valid
	// ...
	return t
}

//  ----------------------------------- BASIC OPS:  -----------------------------------

func leaf(tens *tensor) *node {
	zero_grad := ones(tens.l2, tens.l)
	new_node := node{nil, nil, "leaf", tens, &zero_grad, tens.l2, tens.l}
	return &new_node
}

func add(node1 *node, node2 *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{node1, node2, "+", &data, &zero_grad, l2, l}
	return &new_node
}

func mul(node1 *node, node2 *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{node1, node2, "*", &data, &zero_grad, l2, l}
	return &new_node
}

// matmul(A,x) -> A = [l2 x l] matrix, x = [l] vector
func matmul(a *node, x *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(1, l)
	new_node := node{a, x, "mm", &data, &zero_grad, l2, l}
	return &new_node
}

// PROBLEM: seems like how I am coding it right now, it is assuming that b is already transposed
// // matmul(A,B) -> A = [l2 x l] matrix, b = [] vector
// func matmul2d(a *node, b *node) *node {
// 	l2 = a.l2
// 	l = b.l
// 	zero_grad := ones(l2, l)
// 	data := zeros(1, l)
// 	new_node := node{a, x, "mm", &data, &zero_grad, l2, l}
// 	return &new_node
// }

//  ----------------------------------- ML OPS:  -----------------------------------

func relu(a *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{a, nil, "relu", &data, &zero_grad, l2, l}
	return &new_node
}

func dropout(a *node, l2 int, l int, drop_prob float32) *node {
	drop_prob_data := []float32{drop_prob}
	drop_prob_tensor := tensor{[]*[]float32{&drop_prob_data}, 1, l}
	// drop_prob_tensor := tensornd{last_dim: &drop_prob_data}
	drop_prob_input := leaf(&drop_prob_tensor)
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{a, drop_prob_input, "do", &data, &zero_grad, l2, l}
	return &new_node
}

func linear(in *node, weight *node, bias *node) *node {
	in_times_weight := matmul(weight, in, weight.l2, weight.l)
	if weight.tensor.l2 == 1 || in.tensor.l2 > 1 {
		in_times_weight = mul(weight, in, weight.l2, weight.l)
	}
	plus_bias := add(in_times_weight, bias, in_times_weight.tensor.l2, in_times_weight.l)
	return plus_bias
}

// ----------------------------------- FORWARD AND BACKWARD: -----------------------------------

func forward(root *node) *tensor {
	if root.op == "+" {
		// Return tensor a + b
		t1 := forward(root.left)
		t2 := forward(root.right)
		print(t1.data, t2.data)
		for i := range root.tensor.data {
			t0d := *root.tensor.data[i]
			t1d := *t1.data[i]
			t2d := *t2.data[i]
			for j := range t0d {
				t0d[j] = t1d[j] + t2d[j]
			}
		}
	} else if root.op == "*" {
		// Return tensor a * b
		t1 := forward(root.left)
		t2 := forward(root.right)
		for i := range root.tensor.data {
			t0d := *root.tensor.data[i]
			t1d := *t1.data[i]
			t2d := *t2.data[i]
			for j := range t0d {
				t0d[j] = t1d[j] * t2d[j]
			}
		}
	} else if root.op == "mm" {
		a := forward(root.left)
		x := forward(root.right)
		x_layer := *x.data[0]
		data := zeros(1, root.l)
		data_layer := *data.data[0]
		for i := 0; i < root.l2; i++ {
			a_layer := *a.data[i]
			for j := 0; j < root.l; j++ {
				data_layer[i] += a_layer[j] * x_layer[j]
			}
			root.tensor = &data
		}
	} else if root.op == "relu" {
		a := forward(root.left)
		for i := 0; i < a.l2; i++ {
			a_layer := *a.data[i]
			data_layer := *root.tensor.data[i]
			for j := 0; j < a.l; j++ {
				if a_layer[j] < 0 {
					data_layer[j] = 0
				} else {
					data_layer[j] = a_layer[j]
				}
			}
		}
	} else if root.op == "do" {
		a := forward(root.left)
		for i := 0; i < a.l2; i++ {
			a_layer := *a.data[i]
			data_layer := *root.tensor.data[i]
			drop_prob := *root.right.tensor.data[0]
			drop_prob_float := drop_prob[0]
			for j := 0; j < a.l; j++ {
				if rand.Float32() < drop_prob_float {
					data_layer[j] = 0
				} else {
					data_layer[j] = a_layer[j] / (1 - drop_prob_float) // inverted dropout scaling -> x/(1-p)
				}
			}
		}
	}
	fmt.Println("------------- ", root.op, " -------------")
	if root.left != nil {
		fmt.Println("input 1:")
		for i := range root.left.tensor.data {
			fmt.Println(*root.left.tensor.data[i])
		}
	}
	if root.right != nil {
		fmt.Println("input 2:")
		for i := range root.right.tensor.data {
			fmt.Println(*root.right.tensor.data[i])
		}
	}
	fmt.Println("result:")
	for i := range root.tensor.data {
		fmt.Println(*root.tensor.data[i])
	}
	fmt.Println("")
	return root.tensor
}

func backward(root *node) {
	if root.op == "+" { // add()
		// Chain rule for a + b
		root.left.grad = root.grad
		root.right.grad = root.grad
		backward(root.left)
		backward(root.right)
	} else if root.op == "*" { // mul()
		// Chain rule for a * b
		data1 := zeros(root.l2, root.l)
		for i := 0; i < root.l2; i++ {
			data_layer := *data1.data[i]
			right_layer := *root.right.tensor.data[i]
			grad_layer := *root.grad.data[i]
			for j := 0; j < root.l; j++ {
				data_layer[j] = right_layer[j] * grad_layer[j]
			}
		}
		root.left.grad = &data1

		data2 := zeros(root.l2, root.l)
		for i := 0; i < root.l2; i++ {
			data_layer := *data2.data[i]
			left_layer := *root.left.tensor.data[i]
			grad_layer := *root.grad.data[i]
			for j := 0; j < root.l; j++ {
				data_layer[j] = left_layer[j] * grad_layer[j]
			}
		}
		root.right.grad = &data2
		backward(root.left)
		backward(root.right)
	} else if root.op == "mm" { // matmul()
		// gradient should be 1 x l -> l2 x l
		// a = root.left
		// x = root.right
		right_layer := *root.right.grad.data[0]
		for x := range right_layer {
			right_layer[x] = 0
		}
		root_grad := *root.grad.data[0]
		for i := 0; i < root.l2; i++ {
			left_layer := *root.left.grad.data[i]
			right_data := *root.right.tensor.data[0]
			for j := 0; j < root.l; j++ {
				left_data := *root.left.tensor.data[j]
				left_layer[j] = right_data[j] * root_grad[i]
				right_layer[i] += left_data[i] * root_grad[i]
			}
		}
		backward(root.left)
		backward(root.right)
	} else if root.op == "relu" || root.op == "do" { // relu() or dropout()
		data1 := zeros(root.l2, root.l)
		for i := 0; i < root.l2; i++ {
			data_layer := *data1.data[i]
			relu_layer := *root.tensor.data[i]
			grad_layer := *root.grad.data[i]
			for j := 0; j < root.l; j++ {
				if relu_layer[j] > 0 {
					data_layer[j] = grad_layer[j]
				}
			}
		}
		root.left.grad = &data1
		backward(root.left)
	}
}

// ----------------------------------- TESTING (will be removed when done) -----------------------------------

// Proxy for main() so I can do test runs
func Run() {
	x_0 := []float32{2., -2., 4., 2., -2., 4.}
	x := tensor{[]*[]float32{&x_0}, 1, 6}
	a_0 := []float32{1., -2., 3., 1., -2., 3.}
	a_1 := []float32{-4., 5., 6., -4., 5., 6.}
	a_2 := []float32{7., -8., -9., 7., -8., -9.}
	a_3 := []float32{1., -2., 3., 1., -2., 3.}
	a_4 := []float32{-4., 5., 6., -4., 5., 6.}
	a_5 := []float32{7., -8., -9., 7., -8., -9.}
	a := tensor{[]*[]float32{&a_0, &a_1, &a_2, &a_3, &a_4, &a_5}, 6, 6}
	b_0 := []float32{3., 2., 1., 3., 2., 1.}
	b := tensor{[]*[]float32{&b_0}, 1, 6}
	x_l := leaf(&x)
	a_l := leaf(&a)
	b_l := leaf(&b)
	comp_lookup := map[string]*node{}
	drop := dropout(a_l, a_l.l2, a_l.l, 0.4)
	a_x := matmul(drop, x_l, 6, 6)
	rel := relu(a_x, a_x.tensor.l2, a_x.tensor.l)
	comp_graph := add(rel, b_l, rel.l2, rel.l)
	comp_lookup["x"] = x_l
	comp_lookup["a"] = a_l
	comp_lookup["b"] = b_l
	fmt.Println("Inputs:")
	fmt.Println("x: ", *x.data[0])
	fmt.Println("A: ")
	for i := 0; i < 6; i++ {
		fmt.Println(*a.data[i])
	}
	fmt.Println("b: ", *b.data[0])
	fmt.Println("")
	fmt.Println("Output:")
	fmt.Println("linear 2 = ", *forward(comp_graph).data[0])
	backward(comp_graph)
	fmt.Println("")
	fmt.Println("Gradients: ")
	// fmt.Println("add: ", comp_lookup["add"].grad.data)
	// fmt.Println("mul: ", comp_lookup["mul"].grad.data)
	x_grad := *comp_lookup["x"].grad.data[0]
	fmt.Println("x: ", x_grad)
	fmt.Println("A: ")
	for i := 0; i < 6; i++ {
		fmt.Println(*comp_lookup["a"].grad.data[i])
	}
	b_grad := *comp_lookup["b"].grad.data[0]
	fmt.Println("b: ", b_grad)
}
