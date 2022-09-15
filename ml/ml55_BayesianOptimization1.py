# param_bounds = {'x1 ': (-1,5),
#                 'x2' : (0,4) }

# def y_function(x1, x2):
#     return -x1 **2 - (x2 - 2) **2 + 10

# from bayes_opt import BayesianOptimization

# optimizer = BayesianOptimization(f=y_function, #f = 파라미터를 넣는다.
#                                  pbounds=param_bounds, #파라미터를 딕셔너리 형태로 넣는다.
#                                  random_state=1234)

# optimizer.maximize(init_points=2,
#                    n_iter=20)


#########################################
from bayes_opt import BayesianOptimization

def black_box_function(x, y):
    return -x **2 - (y -2) ** 2 + 10   # ** 제곱

pbounds = {'x' : (-1,5), 'y' : (0, 4)}    # 범위 x: 2~4 / y:-3~3  pbounds = 파라미터

optimizer = BayesianOptimization(
    f = black_box_function,               # f = 모델
    pbounds = pbounds,
    random_state = 1234)

optimizer.maximize(init_points=2,
                   n_iter=20)
