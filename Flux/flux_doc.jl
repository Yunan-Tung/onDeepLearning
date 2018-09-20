using Flux
using Random



Random.seed!(1111);
W1 = param(rand(3, 5))
b1 = param(rand(3))
layer1(x) = W1 * x .+ b1
W2 = param(rand(2, 3))
b2 = param(rand(2))
layer2(x) = W2 * x .+ b2
model(x) = layer2(σ.(layer1(x)))
model(rand(5)) #
#################################################
function linear(in, out)
  ##Random.seed!(1234);
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end

Random.seed!(1111);
linear1 = linear(5, 3) # we can access linear1.W etc
Random.seed!(1111);
linear2 = linear(3, 2)
model(x) = linear2(σ.(linear1(x)))
model(rand(5)) # => 2-element vector
####################################






struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector

##################################################
Random.seed!(1111);
ww1=rand(3,5)
bb1=rand(3)
ww2=rand(2,3)
bb2=rand(2)
mm=rand(5)
W1 = param(ww1)
b1 = param(bb1)
layer1(x) = W1 * x .+ b1
W2 = param(ww2)
b2 = param(bb2)
layer2(x) = W2 * x .+ b2
model(x) = layer2(σ.(layer1(x)))
model(mm) #
#################################################



function linear(in, out)
  Random.seed!(1111);
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end
#Random.seed!(1111);
#linear1 = linear(5, 3) # we can access linear1.W etc
linear1(x) = param(ww1) * x .+ param(bb1)
#Random.seed!(1111);
linear2(x) = param(ww2) * x .+ param(bb2)
model(x) = linear2(σ.(linear1(x)))
model(mm) # => 2-element vector


########################################################

Random.seed!(143134)
rd=rand(10)

model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

model2(rd)



model3 = Chain(
  Dense(10, 5, σ),
 Dense(5, 2),
  softmax
  )

model3(rd)
###################


Wxh = randn(5, 10)
Whh = randn(5, 5)
b   = randn(5)

function rnn(h, x)
  h = tanh.(Wxh * x .+ Whh * h .+ b)
  return h, h
end

x = rand(10) # dummy data
h = rand(5)  # initial hidden state

h, y = rnn(h, x)
