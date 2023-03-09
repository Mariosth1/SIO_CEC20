"""
Created on Sat Mar 14 17:16:52 2020

@author: altzanetos
@modifications : M.Thymianis
"""

import time
from numpy import arange, array
import pandas as pd
from SIO_Alg import SIO

DEBUG = False

"""
main needs to contain the:
    1. pso_iterations = The iterations of PSO in the Search Space.
    2. particles_pop = The population of the particles.
    3. independent_runs = The number of independent test runs of PSO Algorithm.
 """

if __name__ == '__main__':

    func_num = arange(1, 26)
    func_num = func_num[func_num != 24]  # Function No 24 doesn't work

    for j in func_num:

        if DEBUG:
            independent_runs = 1
            generations = 10
            population = 10
        else:
            independent_runs = 40
            NFES = array([[50, 50], [100, 100], [250, 100]])



        if DEBUG:
            Algorithm = SIO(population, generations, j).optimization
            print(j, 'th function', Algorithm(), '\n')

        else:
            for gen_pop in NFES:

                generations = gen_pop[0]
                population = gen_pop[1]

                Algorithm = SIO(population, generations, j).optimization

                start_time = time.time()
                df = pd.DataFrame()
                name_of_file = "SIO_CEC20_F" + str(j) + "_" + str(generations * population) + "FEs.csv"
                for i in range(independent_runs):
                    print(i + 1)
                    [Quality, Penalty, Position, Violation, Exploration, Exploitation, Conv_Iteration
                     ] = Algorithm()
                    # can add Slack depending the benchmark problem
                    temp_df = pd.DataFrame(
                        {
                            'Solution_Quality': Quality,
                            'Penalty': Penalty,
                            'Solution_Allocation': [Position],  # can also add slack columns
                            'Violation': Violation,
                            'AVG_Exploration': Exploration,
                            'AVG_Exploitation': Exploitation,
                            'Conv_Iteration': Conv_Iteration
                        }
                    )

                    df = pd.concat([df, temp_df], ignore_index=False)

                df['Time'] = time.time() - start_time
                "-------------------Change the name of the saved csv-------------------------"
                df.to_csv(name_of_file, sep=',', header=True, index=False)