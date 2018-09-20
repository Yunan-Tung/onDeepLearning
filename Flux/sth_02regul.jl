using Flux: crossentropy
m = Dense(10, 5)
loss(x, y) = crossentropy(softmax(m(x)), y)

#We can regularise this by taking the (L2) norm of the parameters, m.W and m.b.

penalty() = norm(m.W) + norm(m.b)
loss(x, y) = crossentropy(softmax(m(x)), y) + penalty()


 params(m)


##########################

using Flux: onehot, onecold

onehot(:b, [:a, :b, :c])


###################################################
#Here's a larger example with a multi-layer perceptron.

m = Chain(
  Dense(28^2, 128, relu),
  Dense(128, 32, relu),
  Dense(32, 10), softmax)

loss(x, y) = crossentropy(m(x), y) + sum(norm, params(m))

loss(rand(28^2), rand(10))
