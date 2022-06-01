import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random, string, sys


##### handler functions #######
def create_alpha_numeric(length = 16):
    """Creates alpha numeric string with a specific length
    Args:
        length <int> length of alphanumeric string
    Output
        alpha_num <string> string of "length" including alphabets and numerics (e.g. oZMfuT4n2xDglTMJ)
    """
    alpha_num = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))
    return alpha_num

def simulate_execution_time(func, dataframe, sim_num = 2):
    """runs a function with the imported dataframe and returns the execution time of that function
    Args:
        func: <function> the function required to be profiled
        dataframe <DataFrame> dataframe on which func is triggered
        sim_num <int> the number of times function should be run
    Output:
        execution_time <list> execution time of the func on dataframe
    """
    execution_time = []
    for sim in range(sim_num):
        print(sim + 1)
        start = time.time()
        func(df = dataframe)
        end = time.time()
        execution_time.append(round(end - start, 2))
    return execution_time

def pd_concat(df, iter_size = 1000):
    """Implementation using pandas concat method"""
    new_df = df
    for i in range(iter_size):
        new_df = pd.concat([new_df, df.Weight], axis = 1)
    return new_df

def np_concat(df, iter_size = 1000):
    """Implementation using numpy concatenate method"""
    new_df = df
    for i in range(iter_size):
        new_df = np.concatenate([new_df, df.Weight.values.reshape(-1, 1)], axis = 1)
    return new_df


def np_predefined(df, iter_size = 1000):
    """Implementation using numpy 2d array with predefined (pre allocated) memory location"""
    new_df = np.zeros((len(items), iter_size + df.shape[1]))
    for i in range(iter_size):
        new_df[:, i + df.shape[1]] = df.Weight.values
    return new_df

def py_builtin(df, iter_size = 1000):
    """Implementation using python builtin functions and list data structure"""
    new_df = df.values.tolist()
    new_df = np.array(new_df).T.tolist()
    for i in range(1000):
        new_df.append(df.Weight.values.tolist())
    return new_df


if __name__ == '__main__':
    ## get the simulation number from terminal
    sim_num = int(sys.argv[1])
    df_size = 10000
    items = set()
    for i in range(df_size):
        items.add(create_alpha_numeric())
    weights = np.random.randint(5, 1000, len(items))
    df = pd.DataFrame({'Item': list(items), 'Weight': weights})
    pd_concat_execution_time = simulate_execution_time(func = pd_concat, dataframe = df, sim_num = sim_num)
    np_concat_execution_time = simulate_execution_time(func = np_concat, dataframe = df, sim_num = sim_num)
    np_predefined_execution_time = simulate_execution_time(func = np_predefined, dataframe = df, sim_num = sim_num)
    py_builtin_execution_time = simulate_execution_time(func = py_builtin, dataframe = df, sim_num = sim_num)
    
    execution_run = pd.DataFrame(data = {'pandas_concat': pd_concat_execution_time, 
                    'numpy_concatenate': np_concat_execution_time,
                    'numpy_predefined': np_predefined_execution_time,
                    'python_list': py_builtin_execution_time})
    execution_run.to_csv(f'images/execution_time_{sim_num}.csv', index = False)