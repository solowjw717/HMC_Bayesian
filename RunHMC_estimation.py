"""
Created on Feb. 19th  2022
@author: Jingwen Wang
------------------------------------------------------------------------------
This file creates a single Markov Chain for sampling the posterior distributions of model variables.
------------------------------------------------------------------------------
"""
import HamiltonianRTM as HMC
import shelve
import pandas as pd
import os

crop = "Maize"
chain_number = 1
# initilizes HMC experiment from the configuration and observations in the excel file
Exp = HMC.Experiment(crop + '_HMC_Example.xlsx')
for i in range(1,10):
    #the number of sucessive sequences for a single chain (for instance, 10 sequences of an experiment with 8000 iterations will result on 80000 iterations)
    Exp.runExperiment(0)
    if not os.path.exists("output"):
        os.makedirs("output")
    myshelf = shelve.open("output" + os.sep + crop +'_fullemu_chain'+str(chain_number)+'_sequence_'+str(i))
    level0_values = Exp.parameters_value[0] # level 0 3D matrix of dimensions iterations x variables in level 0 x ids in level 0
    level1_values = Exp.parameters_value[1] # level 1 3D matrix of dimensions iterations x variables in level 1 x ids in level 1
    myshelf['level0_values'] = level0_values
    myshelf['level0_variables_order'] = Exp.parameters_Frame.loc[Exp.parameters_Frame['level']==0].sort_values('number_in_level').index.values
    myshelf['level1_values'] = level1_values
    myshelf['level1_variables_order'] = Exp.parameters_Frame.loc[Exp.parameters_Frame['level'] == 1].sort_values('number_in_level').index.values
    myshelf.close()
    level0_values = None
    level1_values = None
    Exp.AddNewIterations() #Creates a new experiment from the last iteration of the current one
