package gograd_nd

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
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
	require_grad bool
}

// Will be implementing an alternative tensor design which uses a single slice instead of slices of pointers of slices of pointers of...
type tensor struct {
	data   []float64 // all tensor data in a single slice
	shape  []int     // [i] = length of dimension i
	diminc []int     // [i] = increment of dimension i
}

//  ----------------------------------- Auxiliary Functions:  -----------------------------------

// NOTE: not currently checking for tensor errors in these functions!!!

func (t *tensor) get(ind []int) float64 {
	index := 0
	for i, x := range ind {
		index += t.diminc[i] * x
	}
	return t.data[index]
}

func (t *tensor) set(ind []int, elem float64) {
	index := 0
	for i, x := range ind {
		index += t.diminc[i] * x
	}
	t.data[index] = elem
}

func (t *tensor) transpose(dim1 int, dim2 int) {
	t.diminc[dim1], t.diminc[dim2] = t.diminc[dim2], t.diminc[dim1]
	t.shape[dim1], t.shape[dim2] = t.shape[dim2], t.shape[dim1]
}

func (t *tensor) l2() int {
	return t.shape[len(t.shape)-2]
}

func (t *tensor) l() int {
	return t.shape[len(t.shape)-1]
}

// Returns a tensor of zeros with the inputted shape
func zeros(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	data := make([]float64, tot_len)
	for i := range data {
		data[i] = 0
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			temp := 1
			for ii := i + 1; ii < len(shape); ii++ {
				temp *= shape[ii]
			}
			di[i] = temp //shape[i+1]
		}
	}
	return tensor{data, shape, di}
}

// Returns a tensor of ones with the inputted shape
func ones(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	data := make([]float64, tot_len)
	for i := range data {
		data[i] = 1
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	return tensor{data, shape, di}
}

func (t *tensor) copy_tens() *tensor {
	copied := make([]float64, len(t.data))
	copy(copied, t.data)
	copied_s := make([]int, len(t.shape))
	copy(copied_s, t.shape)
	copied_i := make([]int, len(t.diminc))
	copy(copied_i, t.diminc)
	return &tensor{copied, copied_s, copied_i}
}

func add_same_size(dst *tensor, t *tensor) {
	for i := range dst.data {
		dst.data[i] += t.data[i]
	}
}

// Initializing for ReLU, Sigmoid, Tanh: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

// Initialize weights with uniform distribution [-1. 1]
func init_weights_uni(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	data := make([]float64, tot_len)
	for i := range data {
		if rand.Intn(2) == 0 {
			data[i] = rand.Float64()
		} else {
			data[i] = -1. * rand.Float64()
		}
	}
	return tensor{data, shape, di}
}

// Initialize weights with normal distribution (0,1)
func init_weights_norm(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	data := make([]float64, tot_len)
	std := math.Sqrt(2. / float64(shape[len(shape)-1]))
	for i := range data {
		data[i] = rand.NormFloat64() * std
	}
	return tensor{data, shape, di}
}

// Initialize weights for ReLU
func init_weights_he(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	data := make([]float64, tot_len)
	std := math.Sqrt(2. / float64(shape[len(shape)-1]))
	for i := range data {
		data[i] = rand.NormFloat64() * std
	}
	return tensor{data, shape, di}
}

// Initialize weights for Sigmoid and TanH
func init_weights_xavier(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	data := make([]float64, tot_len)
	std := math.Sqrt(float64(shape[len(shape)-1]))
	for i := range data {
		if rand.Intn(2) == 0 {
			data[i] = rand.Float64() / std
		} else {
			data[i] = -1. * rand.Float64() / std
		}
	}
	return tensor{data, shape, di}
}

func init_weights_xavier_norm(shape []int) tensor {
	tot_len := 1
	for _, x := range shape {
		tot_len *= x
	}
	di := make([]int, len(shape))
	for i := range shape {
		if i == len(shape)-1 {
			di[i] = 1
		} else {
			di[i] = shape[i+1]
		}
	}
	data := make([]float64, tot_len)
	std := math.Sqrt(6.) / math.Sqrt(float64(shape[len(shape)-1]+shape[len(shape)-2]))
	for i := range data {
		if rand.Intn(2) == 0 {
			data[i] = rand.Float64() * std
		} else {
			data[i] = -1. * rand.Float64() * std
		}
	}
	return tensor{data, shape, di}
}

//  ----------------------------------- BASIC OPS:  -----------------------------------

// Inputs, weights, etc
func leaf(tens *tensor, require_grad bool) *node {
	zero_grad := ones(tens.shape)
	new_node := node{nil, nil, "leaf", tens, &zero_grad, require_grad}
	return &new_node
}

// Syntactic sugar for leaf nodes that are weights
func weight(shape []int, init_type string) *node {
	tens := init_weights_uni(shape)
	if init_type == "uni" {
		tens = init_weights_uni(shape)
	} else if init_type == "norm" {
		tens = init_weights_norm(shape)
	} else if init_type == "he" || init_type == "relu" {
		tens = init_weights_he(shape)
	} else if init_type == "xavier" || init_type == "sigmoid" {
		tens = init_weights_xavier(shape)
	} else if init_type == "xavier norm" {
		tens = init_weights_xavier_norm(shape)
	}
	zero_grad := ones(shape)
	new_node := node{nil, nil, "leaf", &tens, &zero_grad, true}
	return &new_node
}

// Add two identically shaped tensors
func add(node1 *node, node2 *node) *node {
	zero_grad := ones(node1.tensor.shape)
	data := zeros(node1.tensor.shape)
	new_node := node{node1, node2, "+", &data, &zero_grad, true}
	return &new_node
}

// Dot Product
func mul(node1 *node, node2 *node) *node {
	zero_grad := ones(node1.tensor.shape)
	data := zeros(node1.tensor.shape)
	new_node := node{node1, node2, "*", &data, &zero_grad, true}
	return &new_node
}

// Matrix Multiplication
func matmul(a *node, x *node) *node {
	zero_grad := ones([]int{x.tensor.l2(), a.tensor.l2()})
	data := zeros([]int{x.tensor.l2(), a.tensor.l2()})
	new_node := node{a, x, "mm", &data, &zero_grad, true}
	return &new_node
}

//  ----------------------------------- ML OPS:  -----------------------------------

// ReLU layer
func relu(a *node) *node {
	zero_grad := ones(a.tensor.shape)
	data := zeros(a.tensor.shape)
	new_node := node{a, nil, "relu", &data, &zero_grad, true}
	return &new_node
}

// Sigmoid layer
func sigmoid(a *node) *node {
	zero_grad := ones(a.tensor.shape)
	data := zeros(a.tensor.shape)
	new_node := node{a, nil, "sig", &data, &zero_grad, true}
	return &new_node
}

// Dropout layer
func dropout(a *node, drop_prob float64) *node {
	drop_prob_tensor := tensor{[]float64{drop_prob}, []int{1}, []int{1}}
	drop_prob_input := leaf(&drop_prob_tensor, true)
	zero_grad := ones(a.tensor.shape)
	data := zeros(a.tensor.shape)
	new_node := node{a, drop_prob_input, "do", &data, &zero_grad, true}
	return &new_node
}

// Linear layer. Returns output, weight, bias so that weight and bias can be put in params[].
func linear(in *node, out_dim int, weight_type string) (*node, *node, *node) {
	l_weight := weight([]int{out_dim, in.tensor.l()}, weight_type)
	bias := weight([]int{in.tensor.l2(), out_dim}, weight_type)
	in_times_weight := matmul(l_weight, in)
	plus_bias := add(in_times_weight, bias)
	return plus_bias, l_weight, bias
}

// Log Softmax layer
func log_softmax(in *node) *node {
	zero_grad := ones(in.tensor.shape)
	data := zeros(in.tensor.shape)
	new_node := node{in, nil, "sm", &data, &zero_grad, true}
	return &new_node
}

//  ----------------------------------- LOSS AND OPTIM:  -----------------------------------

// Negative Log Likelihood. Takes prediction tensor and target tensor as inputs, returns loss, and updates gradients from input
func nll_loss(pred *tensor, target *tensor, gradients *tensor) float64 {
	loss := 0.
	grad := gradients
	for j := 0; j < target.l2(); j++ {
		for i := 0; i < target.l(); i++ {
			t := target.get([]int{j, i})
			loss -= t * pred.get([]int{j, i})
			grad.set([]int{j, i}, t)
		}
	}

	return loss
}

// initialize adam optimizer
type adam_init struct {
	// b1 float64, b2 float64
	t        float64 // init to 0
	alpha    float64
	prev_m1s []*tensor
	prev_m2s []*tensor
}

// Adam Optimizer Step function
func adam(weights []*node, init adam_init, bsz int) {
	// parameters: would normally get these from adam_init, but keeping them here as defaults for now.
	b1 := 0.9
	b2 := 0.999
	epsilon := math.Pow(10, -8)
	// ------------
	init.t += 1
	t := init.t
	alpha := init.alpha * math.Sqrt((1. - math.Pow(b2, t))) / (1. - math.Pow(b1, t))
	var w_wg sync.WaitGroup
	for k, w := range weights {
		if !w.require_grad {
			continue
		}

		w_wg.Add(1)
		go func(k int, w *node) {
			defer w_wg.Done()
			prev_m1 := *init.prev_m1s[k]
			prev_m2 := *init.prev_m2s[k]
			for i := 0; i < w.grad.l2(); i++ {
				for j := 0; j < w.grad.l(); j++ {
					w.grad.set([]int{i, j}, w.grad.get([]int{i, j})/float64(bsz)) // get mean gradient of batch
					//-gradient--clipping-
					if w.grad.get([]int{i, j}) > 1.0 {
						w.grad.set([]int{i, j}, 1.0)
					}
					if w.grad.get([]int{i, j}) < -1.0 {
						w.grad.set([]int{i, j}, -1.0)
					}
					//-------------------
					m1 := prev_m1.get([]int{i, j})
					m2 := prev_m2.get([]int{i, j})
					biased_m1 := (m1 * b1) + ((1. - b1) * w.grad.get([]int{i, j}))
					biased_m2 := (m2 * b2) + ((1. - b2) * math.Pow(w.grad.get([]int{i, j}), 2.))
					w.tensor.set([]int{i, j}, w.tensor.get([]int{i, j})-(alpha*biased_m1/(math.Sqrt(biased_m2)+epsilon)))
					prev_m1.set([]int{i, j}, biased_m1)
					prev_m2.set([]int{i, j}, biased_m2)
				}
			}
		}(k, w)
	}
	w_wg.Wait()
}

// Exponential learning rate decay
func exp_lr_decay(init adam_init, initial_lr float64, decay_rate float64, global_step int, decay_steps int) {
	init.alpha = initial_lr * math.Pow(decay_rate, float64(global_step/decay_steps))
}

// LR SCHEDULER
func linear_lr_sched(opt adam_init, lr float64, epoch int, reach_max_at int, num_epochs int) {
	if epoch < reach_max_at {
		opt.alpha = opt.alpha + lr/float64(reach_max_at)
	} else {
		opt.alpha = opt.alpha - lr/float64(num_epochs-reach_max_at)
	}
}

// Adaptive LR to prevent plateaus
func adaptive_lr(opt adam_init, total_loss float64, prev_loss float64) {
	if total_loss > prev_loss {
		opt.alpha = opt.alpha * 0.3
	}
	// counterbalance to adaptive LR
	if total_loss < prev_loss {
		opt.alpha = opt.alpha * 1.3 // not sure this is a good idea, but seems like a decent way to speed up training
	}
	// remember to set prev_loss = total_loss after each step
}

// ----------------------------------- FORWARD AND BACKWARD: -----------------------------------

// would probably be more efficient for one data tensor to get passed thru forward instead of each node having its own data tensor.

// Forward Pass function. Recursively finds leaf nodes and gets their values, then performs calculations at each node until it gets back to the root (the output layer).
func forward(root *node) *tensor {
	if root.op == "leaf" {
		return root.tensor
	} else {
		zeroed_data := zeros(root.tensor.shape)
		root.tensor = &zeroed_data
	}
	var wg sync.WaitGroup
	if root.op == "+" { // add
		// Return tensor a + b
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.left)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.right)
		}()
		wg.Wait()
		for i := 0; i < root.tensor.shape[len(root.tensor.shape)-2]; i++ {
			for j := 0; j < root.tensor.shape[len(root.tensor.shape)-1]; j++ {
				root.tensor.set([]int{i, j}, root.left.tensor.get([]int{i, j})+root.right.tensor.get([]int{i, j}))
			}
		}
	} else if root.op == "*" { // mul
		// Return tensor a * b
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.left)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.right)
		}()
		wg.Wait()
		for i := 0; i < root.tensor.shape[len(root.tensor.shape)-2]; i++ {
			for j := 0; j < root.tensor.shape[len(root.tensor.shape)-1]; j++ {
				root.tensor.set([]int{i, j}, root.left.tensor.get([]int{i, j})*root.right.tensor.get([]int{i, j}))
			}
		}
	} else if root.op == "mm" { // matmul
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.left)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward(root.right)
		}()
		wg.Wait()
		a := root.left.tensor
		x := root.right.tensor
		var mm_wg sync.WaitGroup
		for k := 0; k < x.l2(); k++ {
			mm_wg.Add(1)
			go func(k int) {
				defer mm_wg.Done()
				for i := 0; i < a.l2(); i++ {
					for j := 0; j < a.l(); j++ {
						temp2 := (a.get([]int{i, j}) * x.get([]int{k, j}))
						temp := root.tensor.get([]int{k, i}) + temp2
						root.tensor.set([]int{k, i}, temp)
					}
				}
			}(k)
		}
		mm_wg.Wait()
	} else if root.op == "relu" { // relu
		a := forward(root.left)
		for i := 0; i < a.l2(); i++ {
			for j := 0; j < a.l(); j++ {
				if a.get([]int{i, j}) < 0 {
					root.tensor.set([]int{i, j}, 0)
				} else {
					root.tensor.set([]int{i, j}, a.get([]int{i, j}))
				}
			}
		}
	} else if root.op == "do" { // dropout
		a := forward(root.left)
		for i := 0; i < a.l2(); i++ {
			drop_prob_float := root.right.tensor.get([]int{0})
			for j := 0; j < a.l(); j++ {
				if rand.Float64() < drop_prob_float {
					root.tensor.set([]int{i, j}, 0)
				} else {
					root.tensor.set([]int{i, j}, a.get([]int{i, j})/(1-drop_prob_float)) // inverted dropout scaling -> x/(1-p)
				}
			}
		}
	} else if root.op == "sm" { // Softmax
		a := forward(root.left)
		for i := 0; i < a.l2(); i++ {
			layer_sum := 0.
			layer_max := 0.000000001
			// layer_max is for normalization
			for j := 0; j < a.l(); j++ {
				if a.get([]int{i, j}) > layer_max {
					layer_max = a.get([]int{i, j})
				}
			}
			for j := 0; j < a.l(); j++ {
				layer_sum += math.Exp(a.get([]int{i, j}) - layer_max)
			}
			for j := 0; j < a.l(); j++ {
				root.tensor.set([]int{i, j}, a.get([]int{i, j})-math.Log(layer_sum)-layer_max)
			}
		}
	} else if root.op == "sig" { // Sigmoid
		a := forward(root.left)
		for i := 0; i < a.l2(); i++ {
			for j := 0; j < a.l(); j++ {
				root.tensor.set([]int{i, j}, 1./(1.+math.Exp((-1.*a.get([]int{i, j})))))
			}
		}
	}
	return root.tensor
}

// Back Propagation function. Recursively calculates gradients of loss function using chain rule. If zero_grad is true, will set all gradients to zero.
func backward(root *node) {
	var wg sync.WaitGroup
	if root.left != nil {
		zeroed_data := zeros(root.left.grad.shape)
		root.left.grad = &zeroed_data
	}
	if root.right != nil {
		zeroed_data := zeros(root.right.grad.shape)
		root.right.grad = &zeroed_data
	}
	if !root.require_grad {
		return
	} else if root.op == "+" { // add()
		root.left.grad = root.grad.copy_tens() // I dont think I should have to copy (instead could just assign the pointer), but trying this to be safe.
		root.right.grad = root.grad.copy_tens()
		wg.Add(1)
		go func() {
			defer wg.Done()
			backward(root.left)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			backward(root.right)
		}()
		wg.Wait()
	} else if root.op == "*" { // mul()
		// Chain rule for a * b
		for i := 0; i < root.tensor.l2(); i++ {
			for j := 0; j < root.tensor.l(); j++ {
				root.left.grad.set([]int{i, j}, root.right.tensor.get([]int{i, j})*root.grad.get([]int{i, j}))
				root.right.grad.set([]int{i, j}, root.left.tensor.get([]int{i, j})*root.grad.get([]int{i, j}))
			}
		}
		backward(root.left)
		backward(root.right)
	} else if root.op == "mm" { // matmul()
		var mm_wg sync.WaitGroup
		for i := 0; i < root.left.tensor.l2(); i++ {
			mm_wg.Add(1)
			go func(i int) {
				defer mm_wg.Done()
				for i2 := 0; i2 < root.right.tensor.l2(); i2++ {
					for j := 0; j < root.left.tensor.l(); j++ {
						rootgrad := root.grad.get([]int{i2, i})
						t1 := root.left.grad.get([]int{i, j}) + (root.right.tensor.get([]int{i2, j}) * rootgrad)
						t2 := root.right.grad.get([]int{i2, j}) + (root.left.tensor.get([]int{i, j}) * rootgrad)
						root.left.grad.set([]int{i, j}, t1)
						root.right.grad.set([]int{i2, j}, t2)
					}
				}
			}(i)
		}
		mm_wg.Wait()
		wg.Add(1)
		go func() {
			defer wg.Done()
			backward(root.left)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			backward(root.right)
		}()
		wg.Wait()
	} else if root.op == "relu" || root.op == "do" { // relu() or dropout()
		for i := 0; i < root.tensor.l2(); i++ {
			for j := 0; j < root.tensor.l(); j++ {
				if root.tensor.get([]int{i, j}) > 0 {
					root.left.grad.set([]int{i, j}, root.grad.get([]int{i, j}))
				}
			}
		}
		backward(root.left)
	} else if root.op == "sm" {
		for i := 0; i < root.tensor.l2(); i++ {
			for j := 0; j < root.tensor.l(); j++ {
				root.left.grad.set([]int{i, j}, -1.*(root.grad.get([]int{i, j})-math.Exp(root.tensor.get([]int{i, j}))))
			}
		}
		backward(root.left)
	} else if root.op == "sig" {
		for i := 0; i < root.tensor.l2(); i++ {
			for j := 0; j < root.tensor.l(); j++ {
				root.left.grad.set([]int{i, j}, root.tensor.get([]int{i, j})*(1.-root.tensor.get([]int{i, j}))*root.grad.get([]int{i, j}))
			}
		}
		backward(root.left)
	}
}

// ----------------------------------- TESTING (will be removed when done) -----------------------------------
// Example of a Neural Network constructor. Returns output layer node, slice of all parameter leaf nodes, input node, and target node.
func _simple(dim0 int, dim1 int) (*node, []*node, *node) {
	x_placeholder := zeros([]int{dim0, dim1})
	x_node := leaf(&x_placeholder, false)

	// NN:
	l1, l1_weight, l1_bias := linear(x_node, 10, "xavier")
	// s1 := sigmoid(l1)
	// d1 := dropout(s1, 0.1)
	// l2, l2_weight, l2_bias := linear(d1, 64, "xavier")
	// s2 := sigmoid(l2)
	// l25, l25_weight, l25_bias := linear(s2, 128)
	// s25 := sigmoid(l25)
	// d2 := dropout(s2, 0.1)
	// l3, l3_weight, l3_bias := linear(s2, 10, "xavier")
	// s3 := sigmoid(l3)
	// l4, l4_weight, l4_bias := linear(s3, 10, "xavier")
	sm := log_softmax(l1)
	params := []*node{l1_weight, l1_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias, l4_weight, l4_bias}
	// params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias, l4_weight, l4_bias, l25_weight, l25_bias}

	return sm, params, x_node
}

// simple neural net
func Simple() {
	num_batches := 5120 // 51200 // not number of batches, actually just number of samples
	batch_size := 1     //64
	num_epochs := 10
	lr := 0.001 //0.9 //0.99 //0.001

	// Read Data - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
	f, err := os.Open("mnist_train.csv") //fashion-
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	fv, err := os.Open("mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer fv.Close()
	csvReaderv := csv.NewReader(fv)
	datav, err := csvReaderv.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Create train,test sets and val set
	train_x := make([]*tensor, num_batches)
	train_y := make([]*tensor, num_batches)
	test_x := make([]*tensor, num_batches/10.+1) // +1 ensures test set is not empty for train sets w/ len<10
	test_y := make([]*tensor, num_batches/10.+1)
	val_x := make([]*tensor, 10000)
	val_y := make([]*tensor, 10000)
	data_index := 1 // skip first line with column labels
	for batch := range train_x {
		x := zeros([]int{1, 784})
		y := zeros([]int{1, 10})
		for i := 0; i < x.l2(); i++ {
			y_val, _ := strconv.Atoi(data[data_index][0])
			y.set([]int{0, y_val}, 1.)
			for j := 1; j < x.l(); j++ {
				x_val, _ := strconv.Atoi(data[data_index][j])
				x.set([]int{i, j}, float64(x_val)/255.)
			}
			data_index += 1
		}
		train_x[batch] = &x
		train_y[batch] = &y
	}
	for batch := range test_x {
		x := zeros([]int{1, 784})
		y := zeros([]int{1, 10})
		for i := 0; i < x.l2(); i++ {
			y_val, _ := strconv.Atoi(data[data_index][0])
			y.set([]int{0, y_val}, 1.)
			for j := 1; j < x.l(); j++ {
				x_val, _ := strconv.Atoi(data[data_index][j])
				x.set([]int{i, j}, float64(x_val)/255.)
			}
			data_index += 1
		}
		test_x[batch] = &x
		test_y[batch] = &y
	}
	data_index_v := 1 // skip first line with column labels
	for batch := range val_x {
		x := zeros([]int{1, 784})
		y := zeros([]int{1, 10})
		for i := 0; i < x.l2(); i++ {
			y_val, _ := strconv.Atoi(datav[data_index_v][0])
			y.set([]int{0, y_val}, 1.)
			for j := 1; j < x.l(); j++ {
				x_val, _ := strconv.Atoi(datav[data_index_v][j])
				x.set([]int{i, j}, float64(x_val)/255.)
			}
			data_index_v += 1
		}
		val_x[batch] = &x
		val_y[batch] = &y
	}
	loss_list := make([]float64, num_epochs)
	sm, params, x_node := _simple(train_x[0].l2(), train_x[0].l())
	prev_m1s := make([]*tensor, len(params))
	prev_m2s := make([]*tensor, len(params))
	best_loss := 999999999.
	best_epoch := -1
	best_weights := make([]*tensor, len(params))
	batch_gradients := make([]*tensor, len(params))
	step := 0
	for i, x := range params {
		prev1 := zeros(x.grad.shape)
		prev2 := zeros(x.grad.shape)
		bg := zeros(x.grad.shape)
		prev_m1s[i] = &prev1
		prev_m2s[i] = &prev2
		batch_gradients[i] = &bg
	}

	opt := adam_init{0, lr, prev_m1s, prev_m2s}
	fmt.Println("======= START TRAINING ======")
	total_start_time := time.Now()
	for epoch := range loss_list {
		total_loss := 0.
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(train_x), func(i, j int) {
			train_x[i], train_x[j] = train_x[j], train_x[i]
			train_y[i], train_y[j] = train_y[j], train_y[i]
		})
		start_time := time.Now()
		for batch := range train_x {
			x_node.tensor = train_x[batch]
			y := train_y[batch]
			pred := forward(sm)
			zero_grad := zeros(sm.grad.shape)
			sm.grad = &zero_grad
			nll_loss(pred, y, sm.grad)
			// loss := least_squares_loss(pred, y_node.tensor, sm.grad)
			backward(sm)
			for i, x := range params {
				add_same_size(batch_gradients[i], x.grad)
			}
			// minibatching
			if (batch+1)%batch_size == 0 {
				for i, x := range params {
					x.grad = batch_gradients[i].copy_tens()
					bg := zeros(x.grad.shape)
					batch_gradients[i] = &bg
				}
				adam(params, opt, batch_size)
				step += 1
				exp_lr_decay(opt, lr, 0.95, step, num_epochs*batch_size)
			}

			// if batch == num_batches-1 {
			// 	fmt.Println("epoch " + strconv.Itoa(epoch) + " target")
			// 	fmt.Println(y_node.tensor.data)
			// 	fmt.Println("epoch " + strconv.Itoa(epoch) + " preds")
			// 	printar := make([]float64, len(pred.data))
			// 	for i, x := range pred.data {
			// 		printar[i] = math.Exp(x)
			// 	}
			// 	fmt.Println(printar)
			// }
		}
		test_loss := 0.
		for batch := range test_x {
			x_node.tensor = test_x[batch]
			y := test_y[batch]
			pred := forward(sm)
			loss := nll_loss(pred, y, sm.grad)
			test_loss += loss
		}
		total_loss = test_loss / (float64(num_batches))
		end_time := time.Now()
		fmt.Println(epoch, " | ", total_loss, " | ", float64(int(1000*math.Exp(-total_loss)))/1000., " | ", end_time.Sub(start_time))

		if total_loss < best_loss {
			best_loss = total_loss
			best_epoch = epoch
			for i, pnode := range params {
				best_weights[i] = pnode.tensor.copy_tens()
			}
		}
	}
	fmt.Println("BEST LOSS: ", best_epoch, " | ", best_loss)
	fmt.Println("Total training time: ", time.Since(total_start_time))
	for i, pnode := range params {
		pnode.tensor = best_weights[i]
	}
	total_loss := 0.
	correct := 0
	incorrect := 0
	fmt.Println("====== VALIDATING ======")
	for batch := range val_x {
		x_node.tensor = val_x[batch]
		y := val_y[batch]
		pred := forward(sm)

		// get max of pred
		pred_max := 0.
		pred_num := -1
		for i, p := range pred.data {
			pp := math.Exp(p)
			if pp > pred_max {
				pred_max = pp
				pred_num = i
			}
		}
		y_num := -1
		for i, p := range y.data {
			if p > 0 {
				y_num = i
				break
			}
		}
		if y_num == pred_num {
			correct += 1
		} else {
			incorrect += 1
		}
		loss := nll_loss(pred, y, sm.grad)
		// loss := least_squares_loss(pred, y_node.tensor, sm.grad)
		total_loss += loss
	}
	fmt.Println("Validation Data", " | ", total_loss/10000.)
	fmt.Println("Total correct:   ", correct)
	fmt.Println("Total incorrect: ", incorrect)
	fmt.Println(100.*float64(correct)/float64(correct+incorrect), "%")
}

func Test() {
	data := []float64{1, 2, 3, 4}
	dl := []int{2, 2}
	di := []int{2, 1}
	a := tensor{data, dl, di}
	data2 := []float64{1, 2, 3, 4}
	dl2 := []int{2, 2}
	di2 := []int{2, 1}
	b := tensor{data2, dl2, di2}
	add_same_size(&a, &b)
	b = *a.copy_tens()
	a = tensor{data, dl, di}
	fmt.Println(a.data)
	fmt.Println(b.data)
	fmt.Println(data)
	// an := leaf(&a, false)
	// bn := leaf(&b, false)
	// mam := matmul(an, bn)
	// fmt.Println(forward(mam).data)
}

func TransposeTest() {
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
