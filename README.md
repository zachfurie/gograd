
# Autograd in Golang

Gograd_2d: Achieved 98.2% correct predictions on MNIST with an hour of training. Not amazing, but shows that the library works. 

Gograd_nd: Achieved 100% correct predictions on MNIST with 00:01:30 of training. Further exploration required.

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
func _simple(x *tensor, y *tensor) (*node, []*node, *node, *node) {
	x_node := leaf(x, false)
	y_node := leaf(y, false)

	l1, l1_weight, l1_bias := linear(x_node, 128) 
	s1 := sigmoid(l1)
	l2, l2_weight, l2_bias := linear(s1, 64) 
	s2 := sigmoid(l2)
	l3, l3_weight, l3_bias := linear(s2, 10) 
	sm := log_softmax(l3, y_node)
	
	params := []*node{l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias}

	return sm, params, x_node, y_node
}

sm, params, x_node, y_node := _simple(x, y)

batch_gradients := make([]*tensor, len(params))

prev_m1s := make([]*tensor, len(params))
prev_m2s := make([]*tensor, len(params))
opt := adam_init{0, lr, prev_m1s, prev_m2s}

for epoch := range loss_list {
	total_loss := 0.
	for batch := range train_x {
		x_node.tensor = train_x[batch]
		y_node.tensor = train_y[batch]
		pred := forward(sm)
		nll_loss(pred, y_node.tensor, sm.grad)
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
			exp_lr_decay(opt, 0.95, step, num_epochs)
		} else {
			for i, x := range params {
				batch_gradients[i] = add_same_size(batch_gradients[i], x.grad)
			}
		}
	}
}
```
