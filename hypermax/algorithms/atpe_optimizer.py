from .optimization_algorithm_base import OptimizationAlgorithmBase
import hyperopt
import functools
import random
import numpy
import numpy.random
import json
import pkg_resources
from hypermax.hyperparameter import Hyperparameter
import sklearn
import lightgbm
import scipy.stats
import math
from pprint import pprint
import copy
import hypermax.file_utils
from sklearn.cluster import KMeans
from scipy.stats import zscore, f_oneway
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import joblib
import shap

class ATPEOptimizer(OptimizationAlgorithmBase):
    atpeParameters = [
        'gamma',
        'nEICandidates',
        'resultFilteringAgeMultiplier',
        'resultFilteringLossRankMultiplier',
        'resultFilteringMode',
        'resultFilteringRandomProbability',
        # 'clustersQuantile',
        # 'zscoreThreshold',
        'secondaryCorrelationExponent',
        'secondaryCorrelationMultiplier',
        'secondaryCutoff',
        # 'secondaryAnovaExponent',
        # 'secondaryAnovaMultiplier',
        # 'secondaryCatCutoff',
        'secondaryFixedProbability',
        'secondaryLockingMode',
        'secondaryProbabilityMode',
        'secondaryTopLockingPercentile',
    ]

    atpeParameterCascadeOrdering = [
        'resultFilteringMode',
        'secondaryProbabilityMode',
        'secondaryLockingMode',
        'resultFilteringAgeMultiplier',
        'resultFilteringLossRankMultiplier',
        'resultFilteringRandomProbability',
        # 'clustersQuantile',
        # 'zscoreThreshold',
        'secondaryTopLockingPercentile',
        'secondaryCorrelationExponent',
        'secondaryCorrelationMultiplier',
        # 'secondaryAnovaExponent',
        # 'secondaryAnovaMultiplier',
        'secondaryFixedProbability',
        'secondaryCutoff',
        # 'secondaryCatCutoff',
        'gamma',
        'nEICandidates'
    ]

    atpeParameterValues = {
        'resultFilteringMode': ['age', 'loss_rank',
                                #  'cluster', 
                                # 'zscore', 
                                 'none', 'random'],
        'secondaryLockingMode': ['random', 'top'],
        'secondaryProbabilityMode': ['correlation', 'fixed']
    }

    classPredictorKeys = [
        'resultFilteringMode',
        'secondaryLockingMode',
        'secondaryProbabilityMode'
    ]

    atpeModelFeatureKeys = [
        'all_correlation_best_percentile25_ratio',
        'all_correlation_best_percentile50_ratio',
        'all_correlation_best_percentile75_ratio',
        'all_correlation_kurtosis',
        'all_correlation_percentile5_percentile25_ratio',
        'all_correlation_skew',
        'all_correlation_stddev_best_ratio',
        'all_correlation_stddev_median_ratio',
        'all_loss_best_percentile25_ratio',
        'all_loss_best_percentile50_ratio',
        'all_loss_best_percentile75_ratio',
        'all_loss_kurtosis',
        'all_loss_percentile5_percentile25_ratio',
        'all_loss_skew',
        'all_loss_stddev_best_ratio',
        'all_loss_stddev_median_ratio',
        'log10_cardinality',
        'recent_10_correlation_best_percentile25_ratio',
        'recent_10_correlation_best_percentile50_ratio',
        'recent_10_correlation_best_percentile75_ratio',
        'recent_10_correlation_kurtosis',
        'recent_10_correlation_percentile5_percentile25_ratio',
        'recent_10_correlation_skew',
        'recent_10_correlation_stddev_best_ratio',
        'recent_10_correlation_stddev_median_ratio',
        'recent_10_loss_best_percentile25_ratio',
        'recent_10_loss_best_percentile50_ratio',
        'recent_10_loss_best_percentile75_ratio',
        'recent_10_loss_kurtosis',
        'recent_10_loss_percentile5_percentile25_ratio',
        'recent_10_loss_skew',
        'recent_10_loss_stddev_best_ratio',
        'recent_10_loss_stddev_median_ratio',
        'recent_15%_correlation_best_percentile25_ratio',
        'recent_15%_correlation_best_percentile50_ratio',
        'recent_15%_correlation_best_percentile75_ratio',
        'recent_15%_correlation_kurtosis',
        'recent_15%_correlation_percentile5_percentile25_ratio',
        'recent_15%_correlation_skew',
        'recent_15%_correlation_stddev_best_ratio',
        'recent_15%_correlation_stddev_median_ratio',
        'recent_15%_loss_best_percentile25_ratio',
        'recent_15%_loss_best_percentile50_ratio',
        'recent_15%_loss_best_percentile75_ratio',
        'recent_15%_loss_kurtosis',
        'recent_15%_loss_percentile5_percentile25_ratio',
        'recent_15%_loss_skew',
        'recent_15%_loss_stddev_best_ratio',
        'recent_15%_loss_stddev_median_ratio',
        'recent_25_correlation_best_percentile25_ratio',
        'recent_25_correlation_best_percentile50_ratio',
        'recent_25_correlation_best_percentile75_ratio',
        'recent_25_correlation_kurtosis',
        'recent_25_correlation_percentile5_percentile25_ratio',
        'recent_25_correlation_skew',
        'recent_25_correlation_stddev_best_ratio',
        'recent_25_correlation_stddev_median_ratio',
        'recent_25_loss_best_percentile25_ratio',
        'recent_25_loss_best_percentile50_ratio',
        'recent_25_loss_best_percentile75_ratio',
        'recent_25_loss_kurtosis',
        'recent_25_loss_percentile5_percentile25_ratio',
        'recent_25_loss_skew',
        'recent_25_loss_stddev_best_ratio',
        'recent_25_loss_stddev_median_ratio',
        'top_10%_correlation_best_percentile25_ratio',
        'top_10%_correlation_best_percentile50_ratio',
        'top_10%_correlation_best_percentile75_ratio',
        'top_10%_correlation_kurtosis',
        'top_10%_correlation_percentile5_percentile25_ratio',
        'top_10%_correlation_skew',
        'top_10%_correlation_stddev_best_ratio',
        'top_10%_correlation_stddev_median_ratio',
        'top_10%_loss_best_percentile25_ratio',
        'top_10%_loss_best_percentile50_ratio',
        'top_10%_loss_best_percentile75_ratio',
        'top_10%_loss_kurtosis',
        'top_10%_loss_percentile5_percentile25_ratio',
        'top_10%_loss_skew',
        'top_10%_loss_stddev_best_ratio',
        'top_10%_loss_stddev_median_ratio',
        'top_20%_correlation_best_percentile25_ratio',
        'top_20%_correlation_best_percentile50_ratio',
        'top_20%_correlation_best_percentile75_ratio',
        'top_20%_correlation_kurtosis',
        'top_20%_correlation_percentile5_percentile25_ratio',
        'top_20%_correlation_skew',
        'top_20%_correlation_stddev_best_ratio',
        'top_20%_correlation_stddev_median_ratio',
        'top_20%_loss_best_percentile25_ratio',
        'top_20%_loss_best_percentile50_ratio',
        'top_20%_loss_best_percentile75_ratio',
        'top_20%_loss_kurtosis',
        'top_20%_loss_percentile5_percentile25_ratio',
        'top_20%_loss_skew',
        'top_20%_loss_stddev_best_ratio',
        'top_20%_loss_stddev_median_ratio',
        'top_30%_correlation_best_percentile25_ratio',
        'top_30%_correlation_best_percentile50_ratio',
        'top_30%_correlation_best_percentile75_ratio',
        'top_30%_correlation_kurtosis',
        'top_30%_correlation_percentile5_percentile25_ratio',
        'top_30%_correlation_skew',
        'top_30%_correlation_stddev_best_ratio',
        'top_30%_correlation_stddev_median_ratio',
        'top_30%_loss_best_percentile25_ratio',
        'top_30%_loss_best_percentile50_ratio',
        'top_30%_loss_best_percentile75_ratio',
        'top_30%_loss_kurtosis',
        'top_30%_loss_percentile5_percentile25_ratio',
        'top_30%_loss_skew',
        'top_30%_loss_stddev_best_ratio',
        'top_30%_loss_stddev_median_ratio'
    ]

    def __init__(self):
        scalingModelData = json.loads(pkg_resources.resource_string(__name__, "../atpe_models/scaling_model.json"))
        self.featureScalingModels = {}
        for key in self.atpeModelFeatureKeys:
            self.featureScalingModels[key] = sklearn.preprocessing.StandardScaler()
            self.featureScalingModels[key].scale_ = numpy.array(scalingModelData[key]['scales'])
            self.featureScalingModels[key].mean_ = numpy.array(scalingModelData[key]['means'])
            self.featureScalingModels[key].var_ = numpy.array(scalingModelData[key]['variances'])

        self.parameterModels = {}
        self.parameterModelConfigurations = {}
        for param in self.atpeParameters:
            modelData = pkg_resources.resource_string(__name__, "../atpe_models/model-" + param + '.txt')
            # modelData = pkg_resources.resource_string(__name__, "../atpe_models/model-" + param + '.pkl')
            with hypermax.file_utils.ClosedNamedTempFile(modelData) as model_file_name:
                self.parameterModels[param] = lightgbm.Booster(model_file=model_file_name)
                # if param in self.classPredictorKeys:
                #     model = CatBoostClassifier()
                # else:
                #     model = CatBoostRegressor()
                
                # model.load_model(model_file_name)
                # model = joblib.load(model_file_name)
                # self.parameterModels[param] = model

            configString = pkg_resources.resource_string(__name__, "../atpe_models/model-" + param + '-configuration.json')
            # print(configString)
            data = json.loads(configString)
            self.parameterModelConfigurations[param] = data

        self.lastATPEParameters = None
        self.lastLockedParameters = []
        self.atpeParamDetails = None

    def extract_numeric_features(self, obj_dict):
        return [value for key, value in obj_dict.items() if isinstance(value, (int, float))]

    def recommendNextParameters(self, hyperparameterSpace, results, lockedValues=None, paramHistory=[]):
        rstate = numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1)))

        params = {}
        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0.5, 'status': 'ok'}

        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()

        if lockedValues is not None:
            # Remove any locked values from ones the optimizer will examine
            parameters = list(filter(lambda param: param.name not in lockedValues.keys(), parameters))

        log10_cardinality = Hyperparameter(hyperparameterSpace).getLog10Cardinality()
        initializationRounds = max(10, int(log10_cardinality))

        atpeParams = {}
        atpeParamDetails = {}
        # print(results)
        if len(list(result for result in results if result['loss'])) < initializationRounds:
            atpeParams = {
                'gamma': 1.0,
                'nEICandidates': 24,
                'resultFilteringAgeMultiplier': None,
                'resultFilteringLossRankMultiplier': None,
                'resultFilteringMode': "none",
                'resultFilteringRandomProbability': None,
                # 'clustersQuantile': 0.8,
                # 'zscoreThreshold': 0.0,
                'secondaryCorrelationExponent': 1.0,
                'secondaryCorrelationMultiplier': None,
                'secondaryCutoff': 0,
                # 'secondaryAnovaExponent': 1.0,
                # 'secondaryAnovaMultiplier': None,
                # 'secondaryCatCutoff': 0,
                'secondarySorting': 0,
                'secondaryFixedProbability': 0.5,
                'secondaryLockingMode': 'top',
                'secondaryProbabilityMode': 'fixed',
                'secondaryTopLockingPercentile': 0
            }
        else:
            # Calculate the statistics for the distribution
            stats = self.computeAllResultStatistics(hyperparameterSpace, results)
            stats['num_parameters'] = len(parameters)
            stats['log10_cardinality'] = Hyperparameter(hyperparameterSpace).getLog10Cardinality()
            stats['log10_trial'] = math.log10(len(results))
            baseVector = []

            for feature in self.atpeModelFeatureKeys:
                scalingModel = self.featureScalingModels[feature]
                transformed = scalingModel.transform([[stats[feature]]])[0][0]
                baseVector.append(transformed)
          
            baseVector = numpy.array([baseVector])
      
            for atpeParamIndex, atpeParameter in enumerate(self.atpeParameterCascadeOrdering):
                vector = copy.copy(baseVector)[0].tolist()
                # print('base')
            
                # print(len(vector))
                atpeParamFeatures = self.atpeParameterCascadeOrdering[:atpeParamIndex]
                # print(len(atpeParamFeatures))
                # print(atpeParamIndex)
                for atpeParamFeature in atpeParamFeatures:
                    # print(atpeParamFeature)
                    # We have to insert a special value of -3 for any conditional parameters.
                    if atpeParamFeature == 'resultFilteringAgeMultiplier' and atpeParams['resultFilteringMode'] != 'age':
                        # print('resultFilteringAgeMultiplier1')
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'resultFilteringLossRankMultiplier' and atpeParams['resultFilteringMode'] != 'loss_rank':
                        # print('resultFilteringLossRankMultiplier1')
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'resultFilteringRandomProbability' and atpeParams['resultFilteringMode'] != 'random':
                        # print('resultFilteringRandomProbability1')
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryCorrelationMultiplier' and atpeParams['secondaryProbabilityMode'] != 'correlation':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryAnovaMultiplier' and atpeParams['secondaryProbabilityMode'] != 'correlation':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryFixedProbability' and atpeParams['secondaryProbabilityMode'] != 'fixed':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryTopLockingPercentile' and atpeParams['secondaryLockingMode'] != 'top':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    # elif atpeParamFeature == 'clustersQuantile' and atpeParams['resultFilteringMode'] != 'cluster':
                    #     vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    # elif atpeParamFeature == 'zscoreThreshold' and atpeParams['resultFilteringMode'] != 'zscore':
                    #     vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature in self.atpeParameterValues:
                        # print('addin')
                        # print(len(self.atpeParameterValues[atpeParamFeature]))
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            vector.append(1.0 if atpeParams[atpeParamFeature] == value else 0)
                    else:
                        vector.append(float(atpeParams[atpeParamFeature]))
                # print(vector)
                allFeatureKeysForATPEParamModel = copy.copy(self.atpeModelFeatureKeys)
                for atpeParamFeature in atpeParamFeatures:
                    if atpeParamFeature in self.atpeParameterValues:
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            allFeatureKeysForATPEParamModel.append(atpeParamFeature + "_" + value)
                    else:
                        allFeatureKeysForATPEParamModel.append(atpeParamFeature)
                # print(len(vector))
                value = self.parameterModels[atpeParameter].predict([vector], predict_disable_shape_check=True)[0]
                # print(atpeParameter)
                # if atpeParameter in self.classPredictorKeys:
                #     value = self.parameterModels[atpeParameter].predict_proba([vector])[0]
                # else:
                #     value = self.parameterModels[atpeParameter].predict([vector])[0]
                featureContributions = self.parameterModels[atpeParameter].predict([vector], pred_contrib=True, predict_disable_shape_check=True)[0]
                # vector_pool = Pool([vector])
                # featureContributions = self.parameterModels[atpeParameter].get_feature_importance(
                #     data=vector_pool,
                #     type="ShapValues"
                # )[0]
                # For feature contributions (SHAP values)
                # feature_file = pkg_resources.resource_string(__name__, "../atpe_models/features-" + atpeParameter + '.pkl')
                # with hypermax.file_utils.ClosedNamedTempFile(feature_file) as feature_file_name:
                #     features = joblib.load(feature_file_name)
                
                # Reduce the background data size to K samples
                # K = 5  # Choose a value for K, e.g., 100
                # reduced_background = shap.kmeans(features, K)
                # # print(features)
              
                # explainer = shap.KernelExplainer(self.parameterModels[atpeParameter].predict, reduced_background)
                # shap_values = explainer.shap_values(numpy.array([vector]), show_progress=False)

                # In case of multiclass classification, shap_values is a list of arrays (one per class)
                # In case of regression or binary classification, shap_values is a single array
                # featureContributions = shap_values if isinstance(shap_values, list) else shap_values[0]
                # print(featureContributions)
                # print("Feature Contributions Shape:", featureContributions.shape)
                # print("Expected Size:", len(allFeatureKeysForATPEParamModel) + 1, len(self.atpeParameterValues[atpeParameter]))
                
                # print(f"Type of shap_values: {type(shap_values)}")
                # print(f"Length of shap_values (if list): {len(shap_values) if isinstance(shap_values, list) else 'N/A'}")
                # print(f"Shape of shap_values: {shap_values.shape}")
                # If it's a list, check the shape of each element
                # if isinstance(shap_values, list):
                #     for i, class_shap_values in enumerate(shap_values):
                #         print(f"Shape of SHAP values for class {i}: {np.array(class_shap_values).shape}")
                
                # print(value)
                # print(featureContributions)
                atpeParamDetails[atpeParameter] = {
                    "value": None,
                    "reason": None
                }

                # Set the value
                if atpeParameter in self.atpeParameterValues:
                    # Renormalize the predicted probabilities
                    config = self.parameterModelConfigurations[atpeParameter]
                    for atpeParamValueIndex, atpeParamValue in enumerate(self.atpeParameterValues[atpeParameter]):
                        value[atpeParamValueIndex] = (((value[atpeParamValueIndex] - config['predMeans'][atpeParamValue]) / config['predStddevs'][atpeParamValue]) *
                                                      config['origStddevs'][atpeParamValue]) + config['origMeans'][atpeParamValue]
                        value[atpeParamValueIndex] = max(0.0, min(1.0, value[atpeParamValueIndex]))

                    maxVal = numpy.max(value)
                    for atpeParamValueIndex, atpeParamValue in enumerate(self.atpeParameterValues[atpeParameter]):
                        value[atpeParamValueIndex] = max(value[atpeParamValueIndex], maxVal * 0.15)  # We still allow the non reccomended modes to get chosen 15% of the time

                    # Make a random weighted choice based on the normalized probabilities
                    probabilities = value / numpy.sum(value)
                    # print(self.atpeParameterValues[atpeParameter])
                    # print(probabilities)
                    chosen = numpy.random.choice(a=self.atpeParameterValues[atpeParameter], p=probabilities)
                    atpeParams[atpeParameter] = str(chosen)
                else:
                    # Renormalize the predictions
                    config = self.parameterModelConfigurations[atpeParameter]
                    value = (((value - config['predMean']) / config['predStddev']) * config['origStddev']) + config['origMean']
                    atpeParams[atpeParameter] = float(value)

                atpeParamDetails[atpeParameter]["reason"] = {}
                # If we are predicting a class, we get separate feature contributions for each class. Take the average
                if atpeParameter in self.atpeParameterValues:
                    featureContributions = numpy.mean(
                        numpy.reshape(featureContributions, newshape=(len(allFeatureKeysForATPEParamModel) + 1, len(self.atpeParameterValues[atpeParameter]))), axis=1)
                # if isinstance(shap_values, list):
                    # Average the contributions across all classes if needed
                    # featureContributions = numpy.mean([class_shap_values for class_shap_values in shap_values], axis=0)
                # else:
                    # For binary classification or regression, just remove the base value
                    # featureContributions = shap_values[0]
                contributions = [(self.atpeModelFeatureKeys[index], float(featureContributions[index])) for index in range(len(self.atpeModelFeatureKeys))]
                # print(featureContributions)
                # print(contributions)
                
                contributions = sorted(contributions, key=lambda r: -r[1])
                # Only focus on the top 10% of features, since it gives more useful information. Otherwise the total gets really squashed out over many features,
                # because our model is highly regularized.
          
                contributions = contributions[:int(len(contributions) / 10)]
                total = numpy.sum([contrib[1] for contrib in contributions])

                for contributionIndex, contribution in enumerate(contributions[:3]):
                    if total != 0:
                        atpeParamDetails[atpeParameter]['reason'][contribution[0]] = str(int(float(contribution[1]) * 100.0 / total)) + "%"

                # Apply bounds to all the parameters
                if atpeParameter == 'gamma':
                    atpeParams['gamma'] = max(0.2, min(2.0, atpeParams['gamma']))
                if atpeParameter == 'nEICandidates':
                    atpeParams['nEICandidates'] = int(max(2.0, min(48, atpeParams['nEICandidates'])))
                if atpeParameter == 'resultFilteringAgeMultiplier':
                    atpeParams['resultFilteringAgeMultiplier'] = max(1.0, min(4.0, atpeParams['resultFilteringAgeMultiplier']))
                if atpeParameter == 'resultFilteringLossRankMultiplier':
                    atpeParams['resultFilteringLossRankMultiplier'] = max(1.0, min(4.0, atpeParams['resultFilteringLossRankMultiplier']))
                if atpeParameter == 'resultFilteringRandomProbability':
                    atpeParams['resultFilteringRandomProbability'] = max(0.7, min(0.9, atpeParams['resultFilteringRandomProbability']))
                if atpeParameter == 'secondaryCorrelationExponent':
                    atpeParams['secondaryCorrelationExponent'] = max(1.0, min(3.0, atpeParams['secondaryCorrelationExponent']))
                if atpeParameter == 'secondaryCorrelationMultiplier':
                    atpeParams['secondaryCorrelationMultiplier'] = max(0.2, min(1.8, atpeParams['secondaryCorrelationMultiplier']))
                if atpeParameter == 'secondaryCutoff':
                    atpeParams['secondaryCutoff'] = max(-1.0, min(1.0, atpeParams['secondaryCutoff']))
                if atpeParameter == 'secondaryAnovaExponent':
                    atpeParams['secondaryAnovaExponent'] = max(1.0, min(3.0, atpeParams['secondaryAnovaExponent']))
                if atpeParameter == 'secondaryAnovaMultiplier':
                    atpeParams['secondaryAnovaMultiplier'] = max(0.2, min(1.8, atpeParams['secondaryAnovaMultiplier']))    
                if atpeParameter == 'secondaryCatCutoff':
                    atpeParams['secondaryCatCutoff'] = max(-1.0, min(1.0, atpeParams['secondaryCatCutoff']))
                if atpeParameter == 'secondaryFixedProbability':
                    atpeParams['secondaryFixedProbability'] = max(0.2, min(0.8, atpeParams['secondaryFixedProbability']))
                if atpeParameter == 'secondaryTopLockingPercentile':
                    atpeParams['secondaryTopLockingPercentile'] = max(0, min(10.0, atpeParams['secondaryTopLockingPercentile']))
                if atpeParameter == 'clustersQuantile':
                    atpeParams['clustersQuantile'] = max(0.9, min(0.4, atpeParams['clustersQuantile']))
                # if atpeParameter == 'zscoreThreshold':
                #     atpeParams['zscoreThreshold'] = max(-3.0, min(3.0, atpeParams['zscoreThreshold']))

            # Now blank out unneeded params so they don't confuse us
            if atpeParams['secondaryLockingMode'] == 'random':
                atpeParams['secondaryTopLockingPercentile'] = None

            if atpeParams['secondaryProbabilityMode'] == 'fixed':
                atpeParams['secondaryCorrelationMultiplier'] = None
                # atpeParams['secondaryAnovaMultiplier'] = None
            else:
                atpeParams['secondaryFixedProbability'] = None

            if atpeParams['resultFilteringMode'] == 'none':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringLossRankMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
                # atpeParams['clustersQuantile'] = None
                # atpeParams['zscoreThreshold'] = None
            elif atpeParams['resultFilteringMode'] == 'age':
                atpeParams['resultFilteringLossRankMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
                # atpeParams['clustersQuantile'] = None
                # atpeParams['zscoreThreshold'] = None
            elif atpeParams['resultFilteringMode'] == 'loss_rank':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
                # atpeParams['clustersQuantile'] = None
                # atpeParams['zscoreThreshold'] = None
            elif atpeParams['resultFilteringMode'] == 'random':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringLossRankMultiplier'] = None
                # atpeParams['clustersQuantile'] = None
                # atpeParams['zscoreThreshold'] = None
            elif atpeParams['resultFilteringMode'] == 'cluster':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringLossRankMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
                # atpeParams['zscoreThreshold'] = None
            # elif atpeParams['resultFilteringMode'] == 'zscore':
            #     atpeParams['resultFilteringAgeMultiplier'] = None
            #     atpeParams['resultFilteringLossRankMultiplier'] = None
            #     atpeParams['resultFilteringRandomProbability'] = None
                # atpeParams['clustersQuantile'] = None

            for atpeParameter in self.atpeParameters:
                if atpeParams[atpeParameter] is None:
                    del atpeParamDetails[atpeParameter]
                else:
                    atpeParamDetails[atpeParameter]['value'] = atpeParams[atpeParameter]

        self.lastATPEParameters = atpeParams
        self.atpeParamDetails = atpeParamDetails

        # pprint(atpeParams)

        def computePrimarySecondary():
            if len(results) < initializationRounds:
                return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

            if len(set(result['loss'] for result in results)) < 5:
                return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

            numberParameters = [parameter for parameter in parameters if parameter.config['type'] == 'number']
            otherParameters = [parameter for parameter in parameters if parameter.config['type'] != 'number']

            totalWeight = 0
            correlations = {}
            for parameter in numberParameters:
                if len(set(result[parameter.name] for result in results if result[parameter.name] is not None)) < 2:
                    correlations[parameter.name] = 0
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if result[parameter.name] is not None and result['loss'] is not None:
                            values.append(result[parameter.name])
                            valueLosses.append(result['loss'])

                    correlation = math.pow(abs(scipy.stats.spearmanr(values, valueLosses)[0]), atpeParams['secondaryCorrelationExponent'])
                    correlations[parameter.name] = correlation
                    totalWeight += correlation

            # threshold = totalWeight * (1- abs(atpeParams['secondaryCutoff']))
            threshold = totalWeight * abs(atpeParams['secondaryCutoff'])

            if atpeParams['secondaryCutoff'] < 0:
                # Reverse order - we lock in the highest correlated parameters
                sortedParameters = sorted(numberParameters, key=lambda parameter: correlations[parameter.name])
            else:
                # Normal order - sort properties by their correlation to lock in lowest correlated parameters
                sortedParameters = sorted(numberParameters, key=lambda parameter: -correlations[parameter.name])

            primaryParameters = []
            secondaryParameters = []
            cumulative = totalWeight
            for parameter in sortedParameters:
                if cumulative < threshold:
                    secondaryParameters.append(parameter)
                else:
                    primaryParameters.append(parameter)

                cumulative -= correlations[parameter.name]

            return primaryParameters + otherParameters, secondaryParameters, correlations

       
        # def computePrimarySecondary():
        #     if len(results) < initializationRounds:
        #         return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

        #     if len(set(result['loss'] for result in results)) < 5:
        #         return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

        #     numberParameters = [parameter for parameter in parameters if parameter.config['type'] == 'number']
        #     categoricalParameters = [parameter for parameter in parameters if parameter.config['type'] != 'number']

        #     totalWeightNum = 0
        #     totalWeightCat = 0
        #     correlations = {}
        #     categoricalEffects = {}

        #     # Compute Spearman correlation for numerical parameters
        #     for parameter in numberParameters:
        #         if len(set(result[parameter.name] for result in results if result[parameter.name] is not None)) < 2:
        #             correlations[parameter.name] = 0
        #         else:
        #             values = []
        #             valueLosses = []
        #             for result in results:
        #                 if result[parameter.name] is not None and result['loss'] is not None:
        #                     values.append(result[parameter.name])
        #                     valueLosses.append(result['loss'])

        #             correlation = math.pow(abs(scipy.stats.spearmanr(values, valueLosses)[0]), atpeParams['secondaryCorrelationExponent'])
        #             correlations[parameter.name] = correlation
        #             totalWeightNum += correlation

        #     # Compute ANOVA for categorical parameters
        #     for parameter in categoricalParameters:
        #         categories = set(result[parameter.name] for result in results if result[parameter.name] is not None)
        #         if len(categories) < 2:
        #             categoricalEffects[parameter.name] = 0
        #         else:
        #             categoryValues = {category: [] for category in categories}
        #             for result in results:
        #                 if result[parameter.name] is not None and result['loss'] is not None:
        #                     categoryValues[result[parameter.name]].append(result['loss'])

        #             # Compute ANOVA F-statistic
        #             if all(len(values) > 1 for values in categoryValues.values()):
        #                 f_statistic, p_value = f_oneway(*categoryValues.values())
        #                 anova_effect = math.pow(f_statistic, atpeParams['secondaryAnovaExponent'])
        #             else:
        #                 anova_effect = 0

        #             categoricalEffects[parameter.name] = anova_effect
        #             totalWeightCat += anova_effect

        #     # Determine thresholds separately for numerical and categorical parameters
        #     thresholdNum = totalWeightNum * abs(atpeParams['secondaryCutoff'])
        #     thresholdCat = totalWeightCat * (1 - abs(atpeParams['secondaryCatCutoff']))

        #     # Sort numerical parameters by correlation
        #     if atpeParams['secondaryCutoff'] < 0:
        #         sortedNumParameters = sorted(numberParameters, key=lambda parameter: correlations[parameter.name])
        #     else:
        #         sortedNumParameters = sorted(numberParameters, key=lambda parameter: -correlations[parameter.name])

        #     # Sort categorical parameters by ANOVA effect
        #     if atpeParams['secondaryCatCutoff'] < 0:
        #         sortedCatParameters = sorted(categoricalParameters, key=lambda parameter: categoricalEffects[parameter.name])
        #     else:
        #         sortedCatParameters = sorted(categoricalParameters, key=lambda parameter: -categoricalEffects[parameter.name])

        #     # Determine primary and secondary parameters for numerical parameters
        #     primaryParameters = []
        #     secondaryParameters = []
        #     cumulativeNum = totalWeightNum
            
        #     for parameter in sortedNumParameters:
        #         if cumulativeNum < thresholdNum:
        #             secondaryParameters.append(parameter)
        #         else:
        #             primaryParameters.append(parameter)

        #         cumulativeNum -= correlations[parameter.name]

        #     # Determine primary and secondary parameters for categorical parameters
        #     for parameter in sortedCatParameters:
        #         if cumulativeNum < thresholdCat:
        #             secondaryParameters.append(parameter)
        #         else:
        #             primaryParameters.append(parameter)

        #         cumulativeNum -= categoricalEffects[parameter.name]

        #     return primaryParameters, secondaryParameters, {**correlations, **categoricalEffects}

        if len([result['loss'] for result in results if result['loss'] is not None]) == 0:
            maxLoss = 1
        else:
            maxLoss = numpy.max([result['loss'] for result in results if result['loss'] is not None])

        # We create a copy of lockedValues so we don't modify the object that was passed in as an argument - treat it as immutable.
        # The ATPE algorithm will lock additional values in a stochastic manner
        if lockedValues is None:
            lockedValues = {}
        else:
            lockedValues = copy.copy(lockedValues)

        filteredResults = []
        removedResults = []
        if len(results) > initializationRounds:
            primaryParameters, secondaryParameters, correlations = computePrimarySecondary()

            self.lastLockedParameters = []

            sortedResults = list(sorted(list(results), key=lambda result: (result['loss'] if result['loss'] is not None else (maxLoss + 1))))
            topResults = sortedResults
            if atpeParams['secondaryLockingMode'] == 'top':
                topResultsN = max(1, int(math.ceil(len(sortedResults) * atpeParams['secondaryTopLockingPercentile'] / 100.0)))
                topResults = sortedResults[:topResultsN]

            # Any secondary parameters have may be locked to either the current best value or any value within the result pool.
            for secondary in secondaryParameters:
                if atpeParams['secondaryProbabilityMode'] == 'fixed':
                    if random.uniform(0, 1) < atpeParams['secondaryFixedProbability']:
                        self.lastLockedParameters.append(secondary.name)
                        if atpeParams['secondaryLockingMode'] == 'top':
                            lockResult = random.choice(topResults)
                            if lockResult[secondary.name] is not None and lockResult[secondary.name] != "":
                                lockedValues[secondary.name] = lockResult[secondary.name]
                        elif atpeParams['secondaryLockingMode'] == 'random':
                            lockedValues[secondary.name] = self.chooseRandomValueForParameter(secondary)

                elif atpeParams['secondaryProbabilityMode'] == 'correlation':
                    if secondary.config['type'] == 'number':
                        multiplier = atpeParams['secondaryCorrelationMultiplier']
                    else:
                        multiplier = atpeParams['secondaryAnovaMultiplier']
                    probability = max(0, min(1, abs(correlations[secondary.name]) * multiplier))
                    if random.uniform(0, 1) < probability:
                        self.lastLockedParameters.append(secondary.name)
                        if atpeParams['secondaryLockingMode'] == 'top':
                            lockResult = random.choice(topResults)
                            if lockResult[secondary.name] is not None and lockResult[secondary.name] != "":
                                lockedValues[secondary.name] = lockResult[secondary.name]
                        elif atpeParams['secondaryLockingMode'] == 'random':
                            lockedValues[secondary.name] = self.chooseRandomValueForParameter(secondary)

            # if atpeParams['clustersQuantile'] is not None:
            #     clusters_q = atpeParams['clustersQuantile']
            #     numeric_features = [self.extract_numeric_features(obj) for obj in results]

            #     n_clusters = int(round(clusters_q * len(results)))

            #     kmeans = KMeans(n_clusters=int(n_clusters), random_state=0).fit(numeric_features)
            #     labels = kmeans.labels_
            #     # print("Tried clustering with " + str(n_clusters) + " clusters")
            #     # print(labels)
            #     selected_in_cluster = {i: False for i in range(int(n_clusters))}
            
            # if atpeParams['resultFilteringMode'] == 'zscore':
            #     zscores = abs(zscore([result['loss'] for result in results]))

            # Now last step, we filter results prior to sending them into ATPE
            for resultIndex, result in enumerate(results):
                if atpeParams['resultFilteringMode'] == 'none':
                    filteredResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'random':
                    if random.uniform(0, 1) < atpeParams['resultFilteringRandomProbability']:
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'age':
                    age = float(resultIndex) / float(len(results))
                    if random.uniform(0, 1) < (atpeParams['resultFilteringAgeMultiplier'] * age):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'loss_rank':
                    rank = 1.0 - (float(sortedResults.index(result)) / float(len(results)))
                    if random.uniform(0, 1) < (atpeParams['resultFilteringLossRankMultiplier'] * rank):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                # elif atpeParams['resultFilteringMode'] == 'cluster':
                #     if not selected_in_cluster[labels[resultIndex]]:
                #         filteredResults.append(result)
                #         selected_in_cluster[labels[resultIndex]] = True
                #     else:
                #         removedResults.append(result)
                # elif atpeParams['resultFilteringMode'] == 'zscore':
                #     if (atpeParams['zscoreThreshold'] < 0 and zscores[resultIndex] > abs(atpeParams['zscoreThreshold'])) or (atpeParams['zscoreThreshold'] > 0 and zscores[resultIndex] < 3 - atpeParams['zscoreThreshold']):
                #         filteredResults.append(result)
                #     else:
                #         removedResults.append(result)




        # If we are in initialization, or by some other fluke of random nature that we end up with no results after filtering,
        # then just use all the results
        if len(filteredResults) == 0:
            filteredResults = results

        hyperopt.fmin(fn=sample,
                      space=Hyperparameter(hyperparameterSpace).createHyperoptSpace(lockedValues),
                      algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=atpeParams['gamma'],
                                             n_EI_candidates=int(atpeParams['nEICandidates'])),
                      max_evals=1,
                      trials=self.convertResultsToTrials(hyperparameterSpace, filteredResults),
                      rstate=rstate,
                      show_progressbar=False,
                      verbose=False)

        paramHistory.append(atpeParams)
        return params


    def chooseRandomValueForParameter(self, parameter):
        if parameter.config.get('mode', 'uniform') == 'uniform':
            minVal = parameter.config['min']
            maxVal = parameter.config['max']

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                minVal = math.log(minVal)
                maxVal = math.log(maxVal)

            value = random.uniform(minVal, maxVal)

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                value = math.exp(value)

            if 'rounding' in parameter.config:
                value = round(value / parameter.config['rounding']) * parameter.config['rounding']
        elif parameter.get('mode', 'uniform') == 'normal':
            meanVal = parameter.config['mean']
            stddevVal = parameter.config['stddev']

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                meanVal = math.log(meanVal)
                stddevVal = math.log(stddevVal)

            value = random.gauss(meanVal, stddevVal)

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                value = math.exp(value)

            if 'rounding' in parameter.config:
                value = round(value / parameter.config['rounding']) * parameter.config['rounding']
        elif parameter.get('mode', 'uniform') == 'randint':
            max = parameter.config['max']
            value = random.randint(0, max-1)

        return value

    def computePartialResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        bestLoss = 0
        percentile5Loss = 0
        percentile25Loss = 0
        percentile50Loss = 0
        percentile75Loss = 0
        statistics = {}

        numpy.warnings.filterwarnings('ignore')

        if len(set(losses)) > 1:
            bestLoss = numpy.percentile(losses, 0)
            percentile5Loss = numpy.percentile(losses, 5)
            percentile25Loss = numpy.percentile(losses, 25)
            percentile50Loss = numpy.percentile(losses, 50)
            percentile75Loss = numpy.percentile(losses, 75)

            statistics['loss_skew'] = scipy.stats.skew(losses)
            statistics['loss_kurtosis'] = scipy.stats.kurtosis(losses)
        else:
            statistics['loss_skew'] = 0
            statistics['loss_kurtosis'] = 0

        if percentile50Loss == 0:
            statistics['loss_stddev_median_ratio'] = 0
            statistics['loss_best_percentile50_ratio'] = 0
        else:
            statistics['loss_stddev_median_ratio'] = numpy.std(losses) / percentile50Loss
            statistics['loss_best_percentile50_ratio'] = bestLoss / percentile50Loss

        if bestLoss == 0:
            statistics['loss_stddev_best_ratio'] = 0
        else:
            statistics['loss_stddev_best_ratio'] = numpy.std(losses) / bestLoss

        if percentile25Loss == 0:
            statistics['loss_best_percentile25_ratio'] = 0
            statistics['loss_percentile5_percentile25_ratio'] = 0
        else:
            statistics['loss_best_percentile25_ratio'] = bestLoss / percentile25Loss
            statistics['loss_percentile5_percentile25_ratio'] = percentile5Loss / percentile25Loss

        if percentile75Loss == 0:
            statistics['loss_best_percentile75_ratio'] = 0
        else:
            statistics['loss_best_percentile75_ratio'] = bestLoss / percentile75Loss

        def getValue(result, parameter):
            return result[parameter.name]

        # Now we compute correlations between each parameter and the loss
        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()
        correlations = []
        for parameter in parameters:
            if parameter.config['type'] == 'number':
                if len(set(getValue(result, parameter) for result in results if (getValue(result, parameter) is not None and result['loss'] is not None))) < 2:
                    correlations.append(0)
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if result['loss'] is not None and (isinstance(getValue(result, parameter), float) or isinstance(getValue(result, parameter), int)):
                            values.append(getValue(result, parameter))
                            valueLosses.append(result['loss'])

                    correlation = abs(scipy.stats.spearmanr(values, valueLosses)[0])
                    if math.isnan(correlation) or math.isinf(correlation):
                        correlations.append(0)
                    else:
                        correlations.append(correlation)

        correlations = numpy.array(correlations)

        if len(set(correlations)) == 1:
            statistics['correlation_skew'] = 0
            statistics['correlation_kurtosis'] = 0
            statistics['correlation_stddev_median_ratio'] = 0
            statistics['correlation_stddev_best_ratio'] = 0

            statistics['correlation_best_percentile25_ratio'] = 0
            statistics['correlation_best_percentile50_ratio'] = 0
            statistics['correlation_best_percentile75_ratio'] = 0
            statistics['correlation_percentile5_percentile25_ratio'] = 0
        else:
            bestCorrelation = numpy.percentile(correlations, 100)  # Correlations are in the opposite order of losses, higher correlation is considered "best"
            percentile5Correlation = numpy.percentile(correlations, 95)
            percentile25Correlation = numpy.percentile(correlations, 75)
            percentile50Correlation = numpy.percentile(correlations, 50)
            percentile75Correlation = numpy.percentile(correlations, 25)

            statistics['correlation_skew'] = scipy.stats.skew(correlations)
            statistics['correlation_kurtosis'] = scipy.stats.kurtosis(correlations)

            if percentile50Correlation == 0:
                statistics['correlation_stddev_median_ratio'] = 0
                statistics['correlation_best_percentile50_ratio'] = 0
            else:
                statistics['correlation_stddev_median_ratio'] = numpy.std(correlations) / percentile50Correlation
                statistics['correlation_best_percentile50_ratio'] = bestCorrelation / percentile50Correlation

            if bestCorrelation == 0:
                statistics['correlation_stddev_best_ratio'] = 0
            else:
                statistics['correlation_stddev_best_ratio'] = numpy.std(correlations) / bestCorrelation

            if percentile25Correlation == 0:
                statistics['correlation_best_percentile25_ratio'] = 0
                statistics['correlation_percentile5_percentile25_ratio'] = 0
            else:
                statistics['correlation_best_percentile25_ratio'] = bestCorrelation / percentile25Correlation
                statistics['correlation_percentile5_percentile25_ratio'] = percentile5Correlation / percentile25Correlation

            if percentile75Correlation == 0:
                statistics['correlation_best_percentile75_ratio'] = 0
            else:
                statistics['correlation_best_percentile75_ratio'] = bestCorrelation / percentile75Correlation

        return statistics

    def computeAllResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        if len(set(losses)) > 1:
            percentile10Loss = numpy.percentile(losses, 10)
            percentile20Loss = numpy.percentile(losses, 20)
            percentile30Loss = numpy.percentile(losses, 30)
        else:
            percentile10Loss = losses[0]
            percentile20Loss = losses[0]
            percentile30Loss = losses[0]

        allResults = list(results)
        percentile10Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile10Loss]
        percentile20Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile20Loss]
        percentile30Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile30Loss]

        recent10Count = min(len(results), 10)
        recent10Results = results[-recent10Count:]

        recent25Count = min(len(results), 25)
        recent25Results = results[-recent25Count:]

        recent15PercentCount = max(math.ceil(len(results) * 0.15), 5)
        recent15PercentResults = results[-recent15PercentCount:]

        statistics = {}
        allResultStatistics = self.computePartialResultStatistics(hyperparameterSpace, allResults)
        for stat, value in allResultStatistics.items():
            statistics['all_' + stat] = value

        percentile10Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile10Results)
        for stat, value in percentile10Statistics.items():
            statistics['top_10%_' + stat] = value

        percentile20Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile20Results)
        for stat, value in percentile20Statistics.items():
            statistics['top_20%_' + stat] = value

        percentile30Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile30Results)
        for stat, value in percentile30Statistics.items():
            statistics['top_30%_' + stat] = value

        recent10Statistics = self.computePartialResultStatistics(hyperparameterSpace, recent10Results)
        for stat, value in recent10Statistics.items():
            statistics['recent_10_' + stat] = value

        recent25Statistics = self.computePartialResultStatistics(hyperparameterSpace, recent25Results)
        for stat, value in recent25Statistics.items():
            statistics['recent_25_' + stat] = value

        recent15PercentResult = self.computePartialResultStatistics(hyperparameterSpace, recent15PercentResults)
        for stat, value in recent15PercentResult.items():
            statistics['recent_15%_' + stat] = value

        # Although we have added lots of protection in the computePartialResultStatistics code, one last hedge against any NaN or infinity values coming up
        # in our statistics
        for key in statistics.keys():
            if math.isnan(statistics[key]) or math.isinf(statistics[key]):
                statistics[key] = 0

        return statistics