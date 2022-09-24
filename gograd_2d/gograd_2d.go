package gograd_2d

import "fmt"

// Comp graph structs:
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

type tensor1d struct {
	data []float32
	// grad bool
}

type tensornd struct {
	data     []*tensornd
	last_dim *[]float32
	// for all except last dimension, last_dim is nil
	// see shape() for more info
}

// Utilities:

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

// BASIC OPS:
func leaf(tens *tensor) *node {
	zero_grad := ones(tens.l2, tens.l)
	new_node := node{nil, nil, "leaf", tens, &zero_grad, tens.l2, tens.l}
	return &new_node
}

//	func coef(tens *tensor) *node {
//		zero_grad := ones(tens.l2, tens.l)
//		new_node := node{nil, nil, "leaf", tens, &zero_grad, tens.l2, tens.l}
//		return &new_node
//	}
//
//	func input(tens *tensor) *node {
//		zero_grad := ones(tens.l2, tens.l)
//		new_node := node{nil, nil, "leaf", tens, &zero_grad, tens.l2, tens.l}
//		return &new_node
//	}
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

func intermediary(tens *[]float32, node1 *node) *node {
	tens2 := tensor{[]*[]float32{tens}, 1, node1.l}
	new_node := node{node1, nil, "inter", &tens2, node1.grad, node1.l2, node1.l}
	return &new_node
}

// COMPLEX OPS:

// matmul(a,x) -> a = l2xl matrix, x = l vector
func matmul(a *node, x *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	prev := nil
	for i := 0; i < l2; i++ {
		layer := intermediary(a.tensor.data[i], a)
		prev_layer := node{layer, x, "*", &data, &zero_grad, l2, l}
		// sum layer.tensor.data
	}
	new_node := node{node1, node2, "mm", &data, &zero_grad, l2, l}

	return &new_node
}

func relu() {
	return
}

// ML OPS:

func forward(root *node) *tensor {
	if root.left == nil {
		return root.tensor
	} else {
		if root.op == "+" {
			// Return tensor a + b
			t1 := forward(root.left)
			t2 := forward(root.right)
			for i := range root.tensor.data {
				t0d := *root.tensor.data[i]
				t1d := *t1.data[i]
				t2d := *t2.data[i]
				for j := range t0d {
					t0d[j] = t1d[j] + t2d[j]
				}
			}
			return root.tensor
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
			return root.tensor
		}
	}
	return nil
}

func backward(root *node) {
	if root.left == nil {
		return
	} else {
		if root.op == "+" {
			// Chain rule for a + b
			root.left.grad = root.grad
			root.right.grad = root.grad
		} else if root.op == "*" {
			// Chain rule for a * b
			data1 := make([]float32, len(root.right.grad.data))
			for i := range data1 {
				data1[i] = root.right.tensor.data[i] * root.grad.data[i]
			}
			ret1 := tensor{data1}
			root.left.grad = &ret1

			data2 := make([]float32, len(root.left.grad.data))
			for i := range data2 {
				data2[i] = root.left.tensor.data[i] * root.grad.data[i]
			}
			ret2 := tensor{data2}
			root.right.grad = &ret2
		}
		backward(root.left)
		backward(root.right)
	}
}

// NN OPS:
// func linear(in *node, weight *node, bias *node) *node {
// 	in_times_weight := node{in, weight, "*"}
// 	plus_bias := node{&in_times_weight, bias, "+"}
// 	return &plus_bias
// }

func Run() {
	x := tensor{[]float32{0., 1., 2.}}
	a := tensor{[]float32{5., 5., 5.}}
	b := tensor{[]float32{3., 2., 1.}}
	a2 := tensor{[]float32{5., 5., 5.}}
	b2 := tensor{[]float32{3., 2., 1.}}
	x_l := leaf(&x)
	a_l := leaf(&a)
	b_l := leaf(&b)
	a2_l := leaf(&a2)
	b2_l := leaf(&b2)
	comp_lookup := map[string]*node{}
	a_x := mul(x_l, a_l, x_l.length)
	ax_b := add(a_x, b_l, b_l.length)
	comp_graph := add(mul(ax_b, a2_l, a2_l.length), b2_l, b2_l.length)
	comp_lookup["x"] = x_l
	comp_lookup["a"] = a_l
	comp_lookup["b"] = b_l
	comp_lookup["a2"] = a2_l
	comp_lookup["b2"] = b2_l
	comp_lookup["mul"] = a_x
	comp_lookup["add"] = ax_b

	fmt.Println("Layers: ")
	fmt.Println("linear 1 = ax + b")
	fmt.Println("linear 2 = a(linear 1) + b")
	fmt.Println("")
	fmt.Println("Inputs:")
	fmt.Println("x: ", x.data)
	fmt.Println("a: ", a.data)
	fmt.Println("b: ", b.data)
	fmt.Println("")
	fmt.Println("Output:")
	fmt.Println("linear 2 = ", forward(comp_graph).data)
	backward(comp_graph)
	fmt.Println("")
	fmt.Println("Gradients: ")
	// fmt.Println("add: ", comp_lookup["add"].grad.data)
	// fmt.Println("mul: ", comp_lookup["mul"].grad.data)
	fmt.Println("x: ", comp_lookup["x"].grad.data)
	a_grad := ones(len(comp_lookup["a"].grad.data))
	for i := range a_grad.data {
		a_grad.data[i] = comp_lookup["a"].grad.data[i] + comp_lookup["a2"].grad.data[i]
	}
	b_grad := ones(len(comp_lookup["b"].grad.data))
	for i := range b_grad.data {
		b_grad.data[i] = comp_lookup["b"].grad.data[i] + comp_lookup["b2"].grad.data[i]
	}
	fmt.Println("a: ", a_grad.data)
	fmt.Println("b: ", b_grad.data)
	// fmt.Println("a1: ", comp_lookup["a"].grad.data)
	// fmt.Println("b1: ", comp_lookup["b"].grad.data)
	// fmt.Println("a2: ", comp_lookup["a2"].grad.data)
	// fmt.Println("b2: ", comp_lookup["b2"].grad.data)
}
