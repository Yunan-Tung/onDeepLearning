using Flux

m = Chain(x -> x^2, x -> x+1)
m(5) == 26

m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))

#---

d = Dense(5, 2)

d(rand(5))
############################

#chain
using Flux
m = Chain(x -> x^2, x -> x+1)
m(5) == 26  #--->

m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))



d = Dense(5, 2)
d(rand(5))


using Flux
