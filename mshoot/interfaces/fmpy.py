
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fmpy

from fmpy import simulate_fmu, read_model_description, instantiate_fmu, extract, dump

from mshoot import SimModel

class SimFMU(SimModel):

    def __init__(self, fmupath, outputs=None, states=None, parameters=None,
                 verbose=False):
        """
        :param fmupath: str, path to FMU
        :param outputs: list(str), monitored outputs names
        :param states: list(str), monitored states names
        :param parameters: dict, parameters names and values
        :param verbose: bool, whether to suppress pyfmi prints
        """
        if parameters is None:
            parameters = {}
        if states is None:
            states = []
        if outputs is None:
            outputs = []

        self.logger = logging.getLogger(type(self).__name__)
        print(fmupath)
        self.logger.debug("Loading FMU")
        # Load FMU
        model_description = read_model_description(fmupath)
        self.model_description = model_description
        self.unzipdir = extract(fmupath)
        self.fmupath = fmupath
        self.fmu = instantiate_fmu(self.unzipdir, model_description)

        self.outputs = outputs
        self.states = states
        self.parameters = parameters
        self.verbose = verbose

        # Get initial state
        # Comment:
        #   The model has to be initialized to read the state variables.
        #dummy_result = self.fmu.initialize(tStart=0, stopTime=None)
        self.fmu.setupExperiment(startTime=0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.x0 = self._get_state()

        # Reset the FMU
        self.fmu.reset()

        # Set parameters
        #for n in parameters:
         #   self.fmu.set(n, parameters[n])

    def _get_state(self):
        """
        Return an ordered dictionary with state names as keys
        and state values as values.
        """
        # Return dictionary, keys - state names, values - state values
        # get FMU model description object
        #model_description = read_model_description(self.fmupath)
        x = OrderedDict()
        # collect the value references
        # collect the value references
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        # collect list of states and derivatives
        states = []
        #derivatives = []
        for derivative in self.model_description.derivatives:
            #derivatives.append(derivative.variable.name)
            states.append(re.findall('^der\((.*)\)$', derivative.variable.name)[0])

        # collect the value references for states and derivatives
        #vr_states = [vrs[s] for s in states]
        #vr_derivatives = [vrs[x] for x in derivatives]
        for s in states:
            x[s]= self.read(s)# [0] because 1-element array
        return x

    def read(self, datapoint):
        name = self.vrs[datapoint]
        value = self.fmu.getReal([name])
        # print(value)
        return value

    def write(self, datapoint, value):
        name = self.vrs[datapoint]
        self.fmu.setReal([name], [value])

    def simulate(self, udf, x0, save_state=False):
        """
        Simulate the model using the provided inputs `udf`
        and initial state `x0`.
        The DataFrame should have the following content:
        - index - time in seconds and equal steps, named 'time',
        - columns - input data,
        - column names - input variable names.
        The order of `x0` should reflect the one used in `states`.
        Return two DataFrames, `ydf` and `xdf`, with
        outputs and states, respectively, and with the same
        structure as `udf`.
        :param udf: DataFrame, shape (n_steps, n_variables)
        :param x0: vector, size (n_states, )
        :return: ydf, xdf
        """
        assert udf.index.name == 'time'

        timeline = udf.index.values
        start = timeline[0]
        stop = timeline[-1]

        # Prepare inputs for fmpy:
        input_arr = df_to_struct_arr_new(udf)
        assert input_arr is not None, "No inputs assigned"
        output_interval = input_arr[1][0] - input_arr[0][0]

        # Initial condition
        start_values = dict()
        input_names = input_arr.dtype.names
        for name in input_names:
            if name != 'time':
                start_values[name] = input_arr[name][0]

        assert 'time' in input_names, "time must be the first input"

        # Set parameters
        for name, value in self.parameters.items():
            if name != 'time':
                start_values[name] = value

            # Initial states from previous FMU simulation
        for n, value in self.x0.items():
            for n in self.states:
                print (n, value)
                start_values[n] = value

            # Initial states overriden by the user
        i = 0
        for n in self.states:
            start_values[n] = x0[i]
            print(n, x0[i])
            i += 1

        # Simulate
        if not self.verbose:
            nullf = open(os.devnull, 'w')
            sys.stdout = nullf

        self.output_names = list(self.outputs)
        derivative_names = [der.variable.name for der in self.model_description.derivatives]
        # names = [re.search(r'der\((.*)\)', n).group(1) for n in derivative_names]
        for name in derivative_names:
            self.output_names.append(name)

        res = simulate_fmu(
            self.unzipdir,
            start_values=start_values,
            start_time=start,
            stop_time=stop,
            input=input_arr,
            output=self.output_names,
            output_interval=output_interval,
            fmu_instance=self.fmu
            # solver='Euler',  # TODO: It might be useful to add solver/step to options
            # step_size=0.005
        )
        #states = self.fmu.getFMUstate()

        if not self.verbose:
            sys.stdout = sys.__stdout__
            nullf.close()

        # Update state (use only in emulation)
        if save_state:
            self.x0 = self._get_state()

        # Outputs
        res_df = struct_arr_to_df(res)
        t = res['time']

        ydf = pd.DataFrame(index=pd.Index(t, name='time'))
        xdf = pd.DataFrame(index=pd.Index(t, name='time'))

        for n in self.outputs:
            ydf[n] = res[n]

   
        for n in self.states:
            xdf[n] = x0[i]

        self.fmu.reset()

        return ydf, xdf
        
        
if __name__ == "__main__":
    # DEMO: SIMULATE
    # ==============
    # Load FMU
    fmupath = os.path.join('resources', 'fmus', 'R2C2', 'R2C2.fmu')
    parameters = {'C': 1e6}
    model = SimFMU(
        fmupath,
        outputs=['qout', 'Tr'],
        states=['heatCapacitor1.T'],
        parameters=parameters,
        verbose=True)

    # Inputs
    t = np.arange(0, 86401, 3600)
    udf = pd.DataFrame(index=pd.Index(t, name='time'), columns=['q', 'Tout'])
    udf['q'] = np.full(t.size, 100)
    udf['Tout'] = np.full(t.size, 273.15)

    # Initial state
    x0 = [273.15 + 20]

    ydf, xdf = model.simulate(udf, x0)

    ydf.plot(subplots=True, title='ydf')
    xdf.plot(subplots=True, title='xdf')
    plt.show()
