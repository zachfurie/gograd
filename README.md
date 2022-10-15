
# Autograd in Golang

Results:
* Gograd_2d: 
  * Achieved 98.2% correct predictions on MNIST with an hour of training. Not amazing, but shows that the library works. 

* Gograd_nd: 
  * Achieved 100% correct predictions on MNIST with 00:01:24 of training. Further exploration required.
  * Achieved 95% on fashion-mnist with same model/params

To Do:
* log() and exp() for tensors
* max() and mean() for tensors
* reshaping, slicing, etc for tensors
* Conv2d

Long Term:
* print comp graph
* transformer (read Attention Is All You Need)
* rewrite in Rust





## Gograd_2d Example Usage:
```go
// Define Model
func _simple(dim0 int, dim1 int) (*node, []*node, *node) {
	x_placeholder := zeros(dim0, dim1)
	x_node := leaf(x_placeholder, false)
	l1, l1_weight, l1_bias := linear(x_node, 128) 
	s1 := sigmoid(l1)
	l2, l2_weight, l2_bias := linear(s1, 64) 
	s2 := sigmoid(l2)
	l3, l3_weight, l3_bias := linear(s2, 10) 
	out := log_softmax(l3)
	
	params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias}

	return out, params, x_node
}

out, params, x_node := _simple(1, 784) // dimensions of input data

batch_gradients := make([]*tensor, len(params))

prev_m1s := make([]*tensor, len(params))
prev_m2s := make([]*tensor, len(params))
opt := adam_init{0, lr, prev_m1s, prev_m2s}

for epoch := range loss_list {
	total_loss := 0.
	for batch := range train_x {
		x_node.tensor = train_x[batch] // set tensor of input node to next data element
		y := train_y[batch]
		pred := forward(out) // call forward on output node
		nll_loss(pred, y, out.grad) // get loss and pass gradient to output node
		backward(out) // call backward on output node

		// minibatching
		for i, x := range params {
			batch_gradients[i] = add_same_size(batch_gradients[i], x.grad)
		}
		if (batch+1)%batch_size == 0 {
			for i, x := range params {
				x.grad = copy_tens(batch_gradients[i])
				bg := zeros(x.l2, x.l)
				batch_gradients[i] = &bg
			}
			adam(params, opt, batch_size)
			step += 1
			exp_lr_decay(opt, 0.95, step, num_epochs)
		} 
	}
}
```
