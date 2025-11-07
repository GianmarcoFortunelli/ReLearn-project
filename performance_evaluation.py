import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluation(env, agent, test_set):
    static_weight = [0.6, 0.4]
    env.init(test_set)

    n = test_set.shape[1]
    total_value_model = [0.0] * n
    total_value_static = [0.0] * n

    total_value_model[0] = 1.0
    total_value_static[0] = 1.0

    for i in range(1, n):
        static_gain = (
            static_weight[0] * (test_set[1, i] / test_set[1, i-1]) +
            static_weight[1] * (test_set[4, i] / test_set[4, i-1])
        )
        total_value_static[i] = total_value_static[i-1] * static_gain

        status = env.update()
        model_weight = agent.action(status)
        model_gain = (
            model_weight[0] * (test_set[1, i] / test_set[1, i-1]) +
            model_weight[1] * (test_set[4, i] / test_set[4, i-1])
        )
        total_value_model[i] = total_value_model[i-1] * model_gain

    fig, ax = plt.subplots()
    plt.title("Portfolio manager")
    ax.plot(total_value_static, linewidth=2, label="60/40 benchmark")
    ax.plot(total_value_model, linewidth=2, label="RL model")
    ax.legend(fontsize=14)
    plt.show()

    static_overall_gain = total_value_static[-1] - 1
    model_overall_gain = total_value_model[-1] - 1

    print("static overall gain: ", static_overall_gain)
    print("model overall gain: ", model_overall_gain)

    returns_static = np.diff(total_value_static) / np.array(total_value_static[:-1])
    returns_model = np.diff(total_value_model) / np.array(total_value_model[:-1])

    sharpe_static = np.mean(returns_static) / np.std(returns_static, ddof=1)
    sharpe_model = np.mean(returns_model) / np.std(returns_model, ddof=1)

    print("static sharpe ratio: ", sharpe_static)
    print("model sharpe ratio: ", sharpe_model)

    return sharpe_static, sharpe_model