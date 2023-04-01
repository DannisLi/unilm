import nevergrad as ng
import numpy as np

# Similar, but with a noisy case: typically a case in which we train in reinforcement learning.
# This is about parameters rather than hyperparameters. TBPSA is a strong candidate in this case.
# We do *not* manually average over multiple evaluations; the algorithm will take care
# of averaging or reevaluate whatever it wants to reevaluate.


print("Optimization of parameters in reinforcement learning ===============")


def simulate_and_return_test_error_with_rl(x, noisy=True):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x]) + noisy * len(x) * np.random.normal()


budget = 1200  # How many trainings we will do before concluding.


for tool in ["TwoPointsDE", "RandomSearch", "TBPSA", "CMA", "NaiveTBPSA", "NoisyOnePlusOne"]:

    optim = ng.optimizers.registry[tool](parametrization=300, budget=budget)

    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        # Here we ask 3 times and tell 3 times in order to fake asynchronicity
        x1 = optim.ask()
        x2 = optim.ask()
        x3 = optim.ask()
        # The three folowing lines could be parallelized.
        # We could also do things asynchronously, i.e. do one more ask
        # as soon as a training is over.
        y1 = simulate_and_return_test_error_with_rl(*x1.args)
        y2 = simulate_and_return_test_error_with_rl(*x2.args)
        y3 = simulate_and_return_test_error_with_rl(*x3.args)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
        optim.tell(x3, y3)

    recommendation = optim.recommend()
    print("* ", tool, " provides a vector of parameters with test error ",
          simulate_and_return_test_error_with_rl(*recommendation.args, noisy=False))