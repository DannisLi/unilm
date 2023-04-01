import nevergrad as ng

def square(x):
    return sum((x - 0.5) ** 2)

# optimization on x as an array of shape (2,)
optimizer = ng.optimizers.NGOpt(parametrization=5, budget=100)
# recommendation = optimizer.minimize(square)  # best value
# print(recommendation.value)

for i in range(80):
    x = optimizer.ask()
    y = square(*x.args)
    optimizer.tell(x, y)

    recommendation = optimizer.recommend()
    print (i, recommendation.value)