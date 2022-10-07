package gograd_2d

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

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

// 2d tensor
type tensor struct {
	data []*[]float64
	l2   int
	l    int
	// grad bool
}

// nd tensor (will eventually only use this tensor struct)
type tensornd struct {
	data     []*tensornd
	last_dim *[]float64
	l        int // l is length of either data or last_dim, since tensors will only have either one or the other.
	// for all except last dimension, last_dim is nil
	// see shape() for more info
}

// Will be implementing an alternative tensor design which uses a single slice instead of slices of pointers of slices of pointers of...
type tensor1s struct {
	data   []float64 // all tensor data in a single slice
	dimlen []int     // [i] = length of dimension i
	diminc []int     // [i] = increment of dimension i
	l      int       // = len(data)
	dims   int       // = len(dimlen)  = len(diminc)
}

//  ----------------------------------- Auxiliary Functions:  -----------------------------------

// Returns a 2d tensor of size (l2,l)
func zeros(l2 int, l int) tensor {
	zero_grad_pts := make([]*[]float64, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float64, l)
		for j := range zero_grad_data {
			zero_grad_data[j] = 0
		}
		zero_grad_pts[i] = &zero_grad_data
	}
	return tensor{zero_grad_pts, l2, l}
}

// Returns a 2d tensor of size (l2,l)
func ones(l2 int, l int) tensor {
	zero_grad_pts := make([]*[]float64, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float64, l)
		for j := range zero_grad_data {
			zero_grad_data[j] = 1
		}
		zero_grad_pts[i] = &zero_grad_data
	}
	return tensor{zero_grad_pts, l2, l}
}

// Returns a tensor with weights initialized from normal distribution * sqrt(2/n), where n is number of elements being initialized
func init_weights(l2 int, l int) tensor {
	// Initializing for ReLU: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
	std := math.Sqrt(2. / float64(l))
	zero_grad_pts := make([]*[]float64, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float64, l)
		for j := range zero_grad_data {
			if rand.Intn(2) == 0 {
				zero_grad_data[j] = rand.NormFloat64() * std
			} else {
				zero_grad_data[j] = rand.NormFloat64() * std
			}
		}
		zero_grad_pts[i] = &zero_grad_data
	}
	return tensor{zero_grad_pts, l2, l}
}

// Returns tensor of integers in range [-10,10] (used for testing, not really useful otherwise)
func init_ints(l2 int, l int) tensor {
	zero_grad_pts := make([]*[]float64, l2)
	for i := range zero_grad_pts {
		zero_grad_data := make([]float64, l)
		for j := range zero_grad_data {
			if rand.Intn(2) == 0 {
				zero_grad_data[j] = float64(rand.Intn(10))
			} else {
				zero_grad_data[j] = -1. * float64(rand.Intn(10))
			}
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

// sum of float64 slice
func sum(a []float64) float64 {
	result := 0.
	for _, v := range a {
		result += v
	}
	return result
}

// Print Tensor
func pt(t *tensor, name string) {
	fmt.Println("[----- ", name)
	for i := range t.data {
		layer := *t.data[i]
		fmt.Println(i, ": ", layer)
	}
	fmt.Println("-----]")
}

// Print Exp(Tensor)
func pt_exp(t *tensor, name string) {
	fmt.Println("[-----", name)
	for i := range t.data {
		layer := *t.data[i]
		exp_layer := make([]float32, t.l)
		for j := range layer {
			exp_layer[j] = float32(int(1000.*math.Exp(layer[j]))) / 1000.
		}
		fmt.Println(i, ": ", exp_layer)
	}
	fmt.Println("-----]")
}

func copy_tens(t *tensor) *tensor {
	copy_t := zeros(t.l2, t.l)
	for x := range t.data {
		layer := *t.data[x]
		copy_layer := *copy_t.data[x]
		copy(copy_layer, layer)
		// for y := range layer {
		// 	copy_layer[y] = layer[y]
		// }
	}
	return &copy_t
}

func zero_gradients(batch_gradients []*tensor) {
	for i, t := range batch_gradients {
		zero_grad := zeros(t.l2, t.l)
		batch_gradients[i] = &zero_grad
	}
}

func add_same_size(t1 *tensor, t2 *tensor) *tensor {
	ret := zeros(t1.l2, t1.l)
	for i := range ret.data {
		t0d := *ret.data[i]
		t1d := *t1.data[i]
		t2d := *t2.data[i]
		for j := range t0d {
			t0d[j] = t1d[j] + t2d[j]
		}
	}
	return &ret
}

//  ----------------------------------- BASIC OPS:  -----------------------------------

// Inputs, weights, etc
func leaf(tens *tensor, require_grad bool) *node {
	zero_grad := ones(tens.l2, tens.l)
	new_node := node{nil, nil, "leaf", tens, &zero_grad, tens.l2, tens.l, require_grad}
	return &new_node
}

// Syntactic sugar for leaf nodes that are weights
func weight(l2 int, l int) *node {
	tens := init_weights(l2, l)
	zero_grad := ones(tens.l2, tens.l)
	new_node := node{nil, nil, "leaf", &tens, &zero_grad, tens.l2, tens.l, true}
	return &new_node
}

// Add two identically shaped tensors
func add(node1 *node, node2 *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{node1, node2, "+", &data, &zero_grad, l2, l, true}
	return &new_node
}

// Dot Product
func mul(node1 *node, node2 *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{node1, node2, "*", &data, &zero_grad, l2, l, true}
	return &new_node
}

// Matrix Multiplication
func matmul(a *node, x *node, l2 int, l int) *node {
	zero_grad := ones(x.l2, a.l2)
	data := zeros(x.l2, a.l2)
	new_node := node{a, x, "mm", &data, &zero_grad, l2, l, true}
	return &new_node
}

//  ----------------------------------- ML OPS:  -----------------------------------

// ReLU layer
func relu(a *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{a, nil, "relu", &data, &zero_grad, l2, l, true}
	return &new_node
}

// Sigmoid layer
func sigmoid(a *node, l2 int, l int) *node {
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{a, nil, "sig", &data, &zero_grad, l2, l, true}
	return &new_node
}

// Dropout layer
func dropout(a *node, l2 int, l int, drop_prob float64) *node {
	drop_prob_data := []float64{drop_prob}
	drop_prob_tensor := tensor{[]*[]float64{&drop_prob_data}, 1, l}
	drop_prob_input := leaf(&drop_prob_tensor, true)
	zero_grad := ones(l2, l)
	data := zeros(l2, l)
	new_node := node{a, drop_prob_input, "do", &data, &zero_grad, l2, l, true}
	return &new_node
}

// Linear layer. Returns output, weight, bias so that weight and bias can be put in params[].
func linear(in *node, out_dim int) (*node, *node, *node) {
	l_weight := weight(out_dim, in.l)
	bias := weight(in.l2, out_dim)
	in_times_weight := matmul(l_weight, in, l_weight.l2, l_weight.l)
	plus_bias := add(in_times_weight, bias, bias.tensor.l2, bias.tensor.l)
	return plus_bias, l_weight, bias
}

// Log Softmax layer
func log_softmax(in *node, target *node) *node {
	zero_grad := ones(in.tensor.l2, in.tensor.l)
	data := zeros(in.tensor.l2, in.tensor.l)
	new_node := node{in, target, "sm", &data, &zero_grad, in.tensor.l2, in.tensor.l, true}
	return &new_node
}

//  ----------------------------------- LOSS AND OPTIM:  -----------------------------------

// Negative Log Likelihood. Takes prediction tensor and target tensor as inputs, returns loss and gradients
func nll_loss(pred *tensor, target *tensor, gradients *tensor) float64 {
	loss := 0.
	grad := gradients
	for j := range target.data {
		grad_layer := *grad.data[j]
		target_layer := *target.data[j]
		pred_layer := *pred.data[j]
		for i, t := range target_layer {
			loss -= t * pred_layer[i]
			grad_layer[i] -= t
		}
	}

	return loss
}

// Least Squares loss
func least_squares_loss(pred *tensor, target *tensor, gradients *tensor) float64 {
	loss := 0.
	grad := gradients
	for j := range target.data {
		grad_layer := *grad.data[j]
		target_layer := *target.data[j]
		pred_layer := *pred.data[j]
		for i, t := range target_layer {
			loss += math.Pow((pred_layer[i] - t), 2)
			grad_layer[i] = 2 * (pred_layer[i] - t)
		}
	}
	return loss
}

// initialize adam optimizer
type adam_init struct {
	// alpha float32, b1 float32, b2 float32
	t        float64 // init to 0
	alpha    float64 // init to 0.0001
	prev_m1s []*tensor
	prev_m2s []*tensor
}

// Adam Optimizer Step function
func adam(weights []*node, init adam_init, bsz int) {
	// parameters: would normally get these from adam_init
	b1 := 0.9
	b2 := 0.999
	epsilon := math.Pow(10, -8)
	// ------------
	init.t += 1
	t := init.t
	alpha := init.alpha * math.Sqrt((1. - math.Pow(b2, t))) / (1. - math.Pow(b1, t))
	// alpha := init.alpha
	for k, w := range weights {
		if !w.require_grad {
			continue
		}
		prev_m1 := *init.prev_m1s[k]
		prev_m2 := *init.prev_m2s[k]
		for i := 0; i < w.l2; i++ {
			grad_layer := *w.grad.data[i]
			data_layer := *w.tensor.data[i]
			prev_m1_layer := *prev_m1.data[i]
			prev_m2_layer := *prev_m2.data[i]
			for j := 0; j < w.l; j++ {
				grad_layer[j] = grad_layer[j] / float64(bsz) // get mean gradient of batch
				//-gradient--clipping-
				if grad_layer[j] > 1.0 {
					grad_layer[j] = 1.0
				}
				if grad_layer[j] < -1.0 {
					grad_layer[j] = -1.0
				}
				//-------------------
				m1 := prev_m1_layer[j]
				m2 := prev_m2_layer[j]
				biased_m1 := (m1 * b1) + ((1. - b1) * grad_layer[j])
				biased_m2 := (m2 * b2) + ((1. - b2) * math.Pow(grad_layer[j], 2.))

				// m1 = biased_m1 / (1. - math.Pow(b1, t))
				// m2 = biased_m2 / (1. - math.Pow(b2, t))
				// m1 = biased_m1
				// m2 = biased_m2

				data_layer[j] = data_layer[j] - (alpha * biased_m1 / (math.Sqrt(biased_m2) + epsilon))
				prev_m1_layer[j] = biased_m1
				prev_m2_layer[j] = biased_m2
			}
		}
	}
}

// Basic grad descent step
func simple_step(weights []*node, init adam_init) {
	alpha := init.alpha
	for _, w := range weights {
		if !w.require_grad {
			continue
		}
		for i := 0; i < w.l2; i++ {
			grad_layer := *w.grad.data[i]
			data_layer := *w.tensor.data[i]
			for j := 0; j < w.l; j++ {
				data_layer[j] = data_layer[j] - alpha*math.Exp(grad_layer[j])
			}
		}
	}
	init.alpha = alpha * 0.9
}

// Exponential learning rate decay
func exp_lr_decay(init adam_init, decay_rate float64, global_step int, decay_steps int) {
	init.alpha = init.alpha * math.Pow(decay_rate, float64(global_step/decay_steps))
}

// ----------------------------------- FORWARD AND BACKWARD: -----------------------------------

// would probably be more efficient for one data tensor to get passed thru forward instead of each node having its own data tensor.

// Forward Pass function. Recursively finds leaf nodes and gets their values, then performs calculations at each node until it gets back to the root (the output layer).
func forward(root *node) *tensor {
	if root.op == "leaf" {
		return root.tensor
	} else {
		zeroed_data := zeros(root.tensor.l2, root.tensor.l)
		root.tensor = &zeroed_data
	}
	if root.op == "+" { // add
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
	} else if root.op == "*" { // mul
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
	} else if root.op == "mm" { // matmul
		a := forward(root.left)
		x := forward(root.right)
		for k := 0; k < x.l2; k++ {
			x_layer := *x.data[k]
			for i := 0; i < a.l2; i++ {
				a_layer := *a.data[i]
				data_layer := *root.tensor.data[k]
				for j := 0; j < root.l; j++ {
					data_layer[i] += a_layer[j] * x_layer[j]
				}
			}
		}
	} else if root.op == "relu" { // relu
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
	} else if root.op == "do" { // dropout
		a := forward(root.left)
		for i := 0; i < a.l2; i++ {
			a_layer := *a.data[i]
			data_layer := *root.tensor.data[i]
			drop_prob := *root.right.tensor.data[0]
			drop_prob_float := drop_prob[0]
			for j := 0; j < a.l; j++ {
				if rand.Float64() < drop_prob_float {
					data_layer[j] = 0
				} else {
					data_layer[j] = a_layer[j] / (1 - drop_prob_float) // inverted dropout scaling -> x/(1-p)
				}
			}
		}
	} else if root.op == "sm" { // Softmax
		a := forward(root.left)
		for i := 0; i < a.l2; i++ {
			a_layer := *a.data[i]
			data_layer := *root.tensor.data[i]
			layer_sum := 0.
			for j := 0; j < a.l; j++ {
				layer_sum += math.Exp(a_layer[j])
			}
			for j := 0; j < a.l; j++ {
				data_layer[j] = a_layer[j] - math.Log(layer_sum)
			}
		}
	} else if root.op == "sig" { // Sigmoid
		a := forward(root.left)
		for i := 0; i < a.l2; i++ {
			a_layer := *a.data[i]
			data_layer := *root.tensor.data[i]
			for j := 0; j < a.l; j++ {
				data_layer[j] = 1. / (1. + math.Exp((-1. * a_layer[j])))
			}
		}
	}
	return root.tensor
}

// Back Propagation function. Recursively calculates gradients of loss function using chain rule. If zero_grad is true, will set all gradients to zero.
func backward(root *node) {
	if root.left != nil {
		zeroed_data := zeros(root.left.grad.l2, root.left.grad.l)
		root.left.grad = &zeroed_data
	}
	if root.right != nil {
		zeroed_data := zeros(root.right.grad.l2, root.right.grad.l)
		root.right.grad = &zeroed_data
	}
	if !root.require_grad {
		return
	} else if root.op == "+" { // add()
		// Chain rule for a + b
		root.left.grad = copy_tens(root.grad) // I dont think I should have to copy (instead could just assign the pointer), but trying this to be safe.
		root.right.grad = copy_tens(root.grad)
		backward(root.left)
		backward(root.right)
	} else if root.op == "*" { // mul()
		// Chain rule for a * b
		for i := 0; i < root.l2; i++ {
			data_layer := *root.left.grad.data[i]
			right_layer := *root.right.tensor.data[i]
			grad_layer := *root.grad.data[i]
			for j := 0; j < root.l; j++ {
				data_layer[j] = right_layer[j] * grad_layer[j]
			}
		}
		for i := 0; i < root.l2; i++ {
			data_layer := *root.right.grad.data[i]
			left_layer := *root.left.tensor.data[i]
			grad_layer := *root.grad.data[i]
			for j := 0; j < root.l; j++ {
				data_layer[j] = left_layer[j] * grad_layer[j]
			}
		}
		backward(root.left)
		backward(root.right)
	} else if root.op == "mm" { // matmul()
		// gradient should be m x n * n x p -> m x p
		// a = root.left
		// x = root.right
		zero_grad_left := zeros(root.left.tensor.l2, root.left.tensor.l)
		zero_grad_right := zeros(root.right.tensor.l2, root.right.tensor.l)
		root.left.grad = &zero_grad_left
		root.right.grad = &zero_grad_right
		for i := 0; i < root.left.tensor.l2; i++ {
			for i2 := 0; i2 < root.right.tensor.l2; i2++ {
				root_grad := *root.grad.data[i2]
				left_layer := *root.left.grad.data[i]
				right_layer := *root.right.grad.data[i2]
				left_data := *root.left.tensor.data[i]
				right_data := *root.right.tensor.data[i2]
				for j := 0; j < root.left.tensor.l; j++ {
					left_layer[j] += right_data[j] * root_grad[i] // += because element is multiplied by multiple elements in other matrix
					right_layer[j] += left_data[j] * root_grad[i] // += because element is multiplied by multiple elements in other matrix
				}
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
	} else if root.op == "sm" {
		for i := 0; i < root.tensor.l2; i++ {
			grad_layer := *root.grad.data[i]
			// data_layer := *root.tensor.data[i]
			ret_layer := *root.left.grad.data[i]
			left_data_layer := *root.left.tensor.data[i]
			grad_sum := 0.
			for j := 0; j < root.tensor.l; j++ {
				grad_sum += left_data_layer[j] * grad_layer[j]
			}
			for j := 0; j < root.tensor.l; j++ {
				// ret_layer[j] = (1. - data_layer[j]) * grad_layer[j]
				ret_layer[j] = -1. * (left_data_layer[j] - grad_sum) * grad_layer[j]
			}
		}
		backward(root.left)
	} else if root.op == "sig" {
		for i := 0; i < root.tensor.l2; i++ {
			grad_layer := *root.grad.data[i]
			data_layer := *root.tensor.data[i]
			ret_layer := *root.left.grad.data[i]
			for j := 0; j < root.tensor.l; j++ {
				ret_layer[j] = data_layer[j] * (1. - data_layer[j]) * grad_layer[j]
			}
		}
		backward(root.left)
	}
}

// ----------------------------------- TESTING (will be removed when done) -----------------------------------

// Example of a Neural Network constructor. Returns output layer node, slice of all parameter leaf nodes, input node, and target node.
func _simple(x *tensor, y *tensor) (*node, []*node, *node, *node) {
	x_node := leaf(x, false)
	y_node := leaf(y, false)

	// NN:
	l1, l1_weight, l1_bias := linear(x_node, 10) // 784
	// rel1 := relu(l1, x_node.tensor.l2, 5)       // 784
	// l2, l2_weight, l2_bias := linear(sl1, 10) // 256
	// rel2 := relu(l2, x_node.tensor.l2, 128)       // 256
	// l3, l3_weight, l3_bias := linear(rel2, 10)    // 128
	// rel3 := relu(l3, x_node.tensor.l2, 128)
	// l4, l4_weight, l4_bias := linear(rel3, 10)
	sm := log_softmax(l1, y_node)
	// sm := sigmoid(l1, l1.l2, l1.l)
	params := []*node{l1_weight, l1_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias, l4_weight, l4_bias}

	return sm, params, x_node, y_node
}

// simple neural net
func Simple() {
	num_batches := 51200 // 51200 // not number of batches, actually just number of samples
	batch_size := 64
	num_epochs := 100
	// Default learning rate is 0.001, but I have found smaller values to work better throughout my testing
	lr := 0.001 //0.0005 // 0.0001 //0.00001 // 0.000005

	// Read Data - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
	f, err := os.Open("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Create train and test sets
	train_x := make([]*tensor, num_batches)
	train_y := make([]*tensor, num_batches)
	test_x := make([]*tensor, num_batches/10.+1)
	test_y := make([]*tensor, num_batches/10.+1)
	data_index := 5
	for batch := range train_x {
		x := zeros(1, 784)
		y := zeros(1, 10)
		for i := range x.data {
			x_layer := *x.data[i]
			y_layer := *y.data[0]
			y_val, _ := strconv.Atoi(data[data_index][0])
			y_layer[y_val] = 1.
			for j := range x_layer {
				x_val, _ := strconv.Atoi(data[data_index][j])
				x_layer[j] = float64(x_val) / 255.
			}
			data_index += 1
		}
		train_x[batch] = &x
		train_y[batch] = &y
	}
	for batch := range test_x {
		x := zeros(1, 784)
		y := zeros(1, 10)
		for i := range x.data {
			x_layer := *x.data[i]
			y_layer := *y.data[i]
			y_val, _ := strconv.Atoi(data[data_index][0])
			y_layer[y_val] = 1.
			for j := range x_layer {
				x_val, _ := strconv.Atoi(data[data_index][j])
				x_layer[j] = float64(x_val) / 255.
			}
			data_index += 1
		}
		test_x[batch] = &x
		test_y[batch] = &y
	}
	x := train_x[0]
	y := train_x[0]
	loss_list := make([]float64, num_epochs)
	sm, params, x_node, y_node := _simple(x, y)
	prev_m1s := make([]*tensor, len(params))
	prev_m2s := make([]*tensor, len(params))
	best_loss := 999999999.
	// prev_loss := 999999999.
	best_epoch := -1
	best_weights := make([]*tensor, len(params))
	batch_gradients := make([]*tensor, len(params))
	step := 0
	for i, x := range params {
		prev1 := zeros(x.l2, x.l)
		prev2 := zeros(x.l2, x.l)
		bg := zeros(x.l2, x.l)
		prev_m1s[i] = &prev1
		prev_m2s[i] = &prev2
		batch_gradients[i] = &bg
	}

	opt := adam_init{0, lr, prev_m1s, prev_m2s}
	fmt.Println("======= START TRAINING ======")
	for epoch := range loss_list {
		total_loss := 0.
		// LR SCHEDULER
		// if epoch < num_epochs/10 {
		// 	opt.alpha = opt.alpha + lr/float64(num_epochs/10)
		// } else {
		// 	opt.alpha = opt.alpha - lr/float64(9*num_epochs/10)
		// }
		for batch := range train_x {
			x_node.tensor = train_x[batch]
			y_node.tensor = train_y[batch]
			pred := forward(sm)
			loss := 0.

			zero_grad := zeros(sm.grad.l2, sm.grad.l)
			sm.grad = &zero_grad
			loss = nll_loss(pred, y_node.tensor, sm.grad)
			// loss = least_squares_loss(pred, y_node.tensor, sm.grad)
			backward(sm)

			// minibatching
			if (batch+1)%batch_size == 0 {
				for i, x := range params {
					x.grad = copy_tens(batch_gradients[i])
					bg := zeros(x.l2, x.l)
					batch_gradients[i] = &bg
				}
				adam(params, opt, batch_size)
				step += 1
				exp_lr_decay(opt, 0.95, step, 1000)
			} else {
				for i, x := range params {
					batch_gradients[i] = add_same_size(batch_gradients[i], x.grad)
				}
			}
			total_loss += loss

			if batch == num_batches-1 {
				pt(y_node.tensor, "epoch "+strconv.Itoa(epoch)+" target")
				pt_exp(pred, "epoch "+strconv.Itoa(epoch)+" preds")
			}
		}

		total_loss = total_loss / (float64(num_batches))
		fmt.Println(epoch, " | ", total_loss, " | ", float64(int(1000*math.Exp(-total_loss)))/1000., " | ", time.Now())

		// Adaptive LR to prevent plateaus
		// if total_loss > prev_loss {
		// 	opt.alpha = opt.alpha * 0.3
		// }

		if total_loss < best_loss {
			best_loss = total_loss
			best_epoch = epoch
			for i, pnode := range params {
				best_weights[i] = copy_tens(pnode.tensor)
			}
		}
		// counterbalance to adaptive LR
		// if total_loss < prev_loss {
		// 	opt.alpha = opt.alpha * 1.3 // not sure this is a good idea, but seems like a decent way to speed up training
		// }
		// prev_loss = total_loss
	}
	fmt.Println("BEST LOSS: ", best_epoch, " | ", best_loss)
	for i, pnode := range params {
		pnode.tensor = best_weights[i]
	}
	total_loss := 0.
	correct := 0
	incorrect := 0
	fmt.Println("====== VALIDATING ======")
	for batch := range test_x {
		x_node.tensor = test_x[batch]
		y_node.tensor = test_y[batch]
		pred := forward(sm)

		// get max of pred
		pred_layer := *pred.data[0]
		pred_max := 0.
		pred_num := -1
		for i, p := range pred_layer {
			pp := math.Exp(p)
			if pp > pred_max {
				pred_max = pp
				pred_num = i
			}
		}
		y_layer := *y_node.tensor.data[0]
		y_num := -1
		for i, p := range y_layer {
			if p > 0 {
				y_num = i
				break
			}
		}
		// fmt.Println("y value: ", y_num, " | prediction: ", pred_num)
		if y_num == pred_num {
			correct += 1
		} else {
			incorrect += 1
		}
		loss := nll_loss(pred, y_node.tensor, sm.grad)
		total_loss += loss
	}
	fmt.Println("Validation Data", " | ", total_loss/(float64(len(test_x))))
	fmt.Println("Total correct:   ", correct)
	fmt.Println("Total incorrect: ", incorrect)
	fmt.Println(100.*correct/(correct+incorrect), "%")
}
