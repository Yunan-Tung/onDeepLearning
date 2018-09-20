using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3


#-----------------------------

using Flux.Tracker

W = param(W)
b = param(b)

gs = Tracker.gradient(() -> loss(x, y), Params([W, b]))


using Flux.Tracker: update!

Δ = gs[W]

# Update the parameter and reset the gradient
update!(W, -0.1Δ)

loss(x, y) # ~ 2.5

#################################################

#Building Layers
W1 = param(rand(3, 5))
b1 = param(rand(3))
layer1(x) = W1 * x .+ b1

W2 = param(rand(2, 3))
b2 = param(rand(2))
layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5)) # => 2-element vector


############################################################

dd = Dense(5, 2)
#xx=rand(5)
xx= [1 2 3 4 5]

dd(rand(5))

#####################################3

using Flux

d = Dense(5, 2)
d(rand(5))


d = Dense(3,1)
d(rand(3))


#############################
struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)))

a = Affine(10, 5)

a(rand(10))
