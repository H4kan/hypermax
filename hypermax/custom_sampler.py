# from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler
# import numpy as np

# class CustomMOTPEMultiObjectiveSampler(MOTPEMultiObjectiveSampler):
#     def __init__(self, initial_params, *args, **kwargs):
#         self.atpeParams = initial_params
#         super().__init__(*args, **kwargs)

#     def sample_independent(self, study, trial, param_name, param_distribution):
#         # Dynamically adjust gamma or other parameters based on trial count
#         n_trials = len(study.trials)

#         # Example logic: adjust parameters after a certain number of trials
#         if n_trials > 10:
#             self.gamma = lambda n: max(1, int(np.ceil(self.atpeParams['gamma'] * n)))
#             self.n_ehvi_candidates = int(self.atpeParams['nEICandidates'])

#         # Call the original sampling method
#         return super().sample_independent(study, trial, param_name, param_distribution)

from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler
from optuna.multi_objective.samplers._motpe import MOTPESampler, _create_study, _create_trial
from optuna.samplers import TPESampler
from typing import Callable, Optional
import numpy as np

class CustomMOTPESampler(MOTPESampler):
    def __init__(self, atpeParams, *args, **kwargs):
        self.atpeParams = atpeParams
        super().__init__(*args, **kwargs)
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        # Dynamically update parameters before sampling
        n_ehvi_candidates = self.atpeParams.get('nEICandidates', 24)
        gamma_value = self.atpeParams.get('gamma', 0.1)

        # Set the dynamic parameters into the sampler before sampling
        self._n_ei_candidates = n_ehvi_candidates
        self._gamma = lambda n: max(1, int(np.ceil(gamma_value * n)))

        # Perform sampling as usual
        return super().sample_independent(study, trial, param_name, param_distribution)


class CustomMOTPEMultiObjectiveSampler(MOTPEMultiObjectiveSampler):
    def __init__(self, atpeParams, *args, **kwargs):
        # Use the custom MOTPESampler instead of the default MOTPESampler
        self.atpeParams = atpeParams
        super().__init__(*args, **kwargs)
        self._motpe_sampler = CustomMOTPESampler(atpeParams, *args, **kwargs)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Delegate to the custom MOTPESampler which handles the dynamic updates
        return self._motpe_sampler.sample_independent(
            _create_study(study),
            _create_trial(trial),
            param_name,
            param_distribution
        )
