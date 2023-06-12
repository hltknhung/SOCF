import os

import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
# from neupy import algorithms
from pyGRNN import GRNN
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import TuningParameter
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from WithoutClustering import CONSTANT

def run_cases(df_train, df_test, basedir):
    df_train['x1_tcf'] = df_train['T1'].astype(float) * 2
    df_train['x2_tcf'] = df_train['T3'].astype(float) * 1
    df_train['x3_tcf'] = df_train['T4'].astype(float) * 1
    df_train['x4_tcf'] = df_train['T5'].astype(float) * 1
    df_train['x5_tcf'] = df_train['T6'].astype(float) * 0.5
    df_train['x6_tcf'] = df_train['T7'].astype(float) * 0.5
    df_train['x7_tcf'] = df_train['T9'].astype(float) * 1
    df_train['x8_tcf'] = df_train['T10'].astype(float) * 1
    df_train['x9_tcf'] = df_train['T11'].astype(float) * 1
    X_train_tcf = df_train[['x1_tcf', 'x2_tcf', 'x3_tcf', 'x4_tcf', 'x5_tcf', 'x6_tcf', 'x7_tcf', 'x8_tcf', 'x9_tcf']]
    y_train_tcf = df_train['TCF'].astype(float)
    regression_tcf = linear_model.LinearRegression(fit_intercept=False)
    regression_tcf.fit(X_train_tcf, y_train_tcf)
    df_train['LaTF'] = regression_tcf.predict(X_train_tcf)

    df_train['x1_ecf'] = df_train['ENV2'].astype(float) * 0.5
    df_train['x2_ecf'] = df_train['ENV4'].astype(float) * 0.5
    df_train['x3_ecf'] = df_train['ENV5'].astype(float) * 1
    df_train['x4_ecf'] = df_train['ENV6'].astype(float) * 2
    df_train['x5_ecf'] = df_train['ENV7'].astype(float) * -1
    df_train['x6_ecf'] = df_train['ENV8'].astype(float) * 2
    X_train_ecf = df_train[['x1_ecf', 'x2_ecf', 'x3_ecf', 'x4_ecf', 'x5_ecf', 'x6_ecf']]
    y_train_ecf = df_train['ECF'].astype(float)
    regression_ecf = linear_model.LinearRegression(fit_intercept=False)
    regression_ecf.fit(X_train_ecf, y_train_ecf)
    df_train['LaEF'] = regression_ecf.predict(X_train_ecf)

    df_test['x1_tcf'] = df_test['T1'].astype(float) * 2
    df_test['x2_tcf'] = df_test['T3'].astype(float) * 1
    df_test['x3_tcf'] = df_test['T4'].astype(float) * 1
    df_test['x4_tcf'] = df_test['T5'].astype(float) * 1
    df_test['x5_tcf'] = df_test['T6'].astype(float) * 0.5
    df_test['x6_tcf'] = df_test['T7'].astype(float) * 0.5
    df_test['x7_tcf'] = df_test['T9'].astype(float) * 1
    df_test['x8_tcf'] = df_test['T10'].astype(float) * 1
    df_test['x9_tcf'] = df_test['T11'].astype(float) * 1

    X_test_tcf = df_test[['x1_tcf', 'x2_tcf', 'x3_tcf', 'x4_tcf', 'x5_tcf', 'x6_tcf', 'x7_tcf', 'x8_tcf', 'x9_tcf']]
    y_test_tcf = df_test['TCF']
    laTF = regression_tcf.predict(X_test_tcf)
    df_test['LaTF'] = laTF.round(3)

    df_test['x1_ecf'] = df_test['ENV2'].astype(float) * 0.5
    df_test['x2_ecf'] = df_test['ENV4'].astype(float) * 0.5
    df_test['x3_ecf'] = df_test['ENV5'].astype(float) * 1
    df_test['x4_ecf'] = df_test['ENV6'].astype(float) * 2
    df_test['x5_ecf'] = df_test['ENV7'].astype(float) * -1
    df_test['x6_ecf'] = df_test['ENV8'].astype(float) * 2

    X_test_ecf = df_test[['x1_ecf', 'x2_ecf', 'x3_ecf', 'x4_ecf', 'x5_ecf', 'x6_ecf']]
    y_test_ecf = df_test['ECF'].astype(float)
    laEF = regression_ecf.predict(X_test_ecf)
    df_test['LaEF'] = laEF.round(3)
    df_train['OCF'] = (df_train['UAW'].astype(float) + df_train['UUCW'].astype(float)) \
                      * df_train['LaEF'].astype(float) * df_train['LaTF'].astype(float)
    df_test['OCF'] = (df_test['UAW'].astype(float) + df_test['UUCW'].astype(float)) \
                     * df_test['LaEF'].astype(float) * df_test['LaTF'].astype(float)

    df_train['x1_OCF'] = df_train['UAW'].astype(float) * df_train['LaTF'].astype(float) * df_train['LaEF'].astype(float)
    df_train['x2_OCF'] = df_train['UUCW'].astype(float) * df_train['LaTF'].astype(float) * df_train['LaEF'].astype(
        float)
    X_train_OCF = df_train[['x1_OCF', 'x2_OCF']]
    y_train_OCF = df_train['Real_P20'].astype(float)
    df_test['x1_OCF'] = df_test['UAW'].astype(float) * df_test['LaTF'].astype(float) * df_test['LaEF'].astype(float)
    df_test['x2_OCF'] = df_test['UUCW'].astype(float) * df_test['LaTF'].astype(float) * df_test['LaEF'].astype(float)
    X_test_OCF = df_test[['x1_OCF', 'x2_OCF']]

    # OCF&MLR
    MLR_Reg = linear_model.LinearRegression(fit_intercept=True)
    MLR_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['MLR'] = MLR_Reg.predict(X_train_OCF)
    MLR_result = MLR_Reg.predict(X_test_OCF)
    df_test['MLR'] = MLR_result.round(3)

    # OCF&SVR
    bestParaSVR = TuningParameter.tuningSVR(X_train_OCF, y_train_OCF)
    SVR_Reg = SVR(bestParaSVR)
    SVR_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['SVR'] = SVR_Reg.predict(X_train_OCF)
    SVR_result = SVR_Reg.predict(X_test_OCF)
    df_test['SVR'] = SVR_result.round(3)

    # OCF&MLP
    bestParaMLP = TuningParameter.tuningMLP(X_train_OCF, y_train_OCF)
    MLP_Reg = MLPRegressor(bestParaMLP)
    MLP_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['MLP'] = MLP_Reg.predict(X_train_OCF)
    MLP_result = MLP_Reg.predict(X_test_OCF)
    df_test['MLP'] = MLP_result.round(3)

    # OCF&GB
    bestParaGB = TuningParameter.tuningGB(X_train_OCF, y_train_OCF)
    GB_Reg = GradientBoostingRegressor(bestParaGB)
    GB_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['GB'] = GB_Reg.predict(X_train_OCF)
    GB_result = GB_Reg.predict(X_test_OCF)
    df_test['GB'] = GB_result.round(3)

    # OCF&RF
    bestParaRF = TuningParameter.tuningRF(X_train_OCF, y_train_OCF)
    RF_Reg = RandomForestRegressor(bestParaRF)
    RF_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['RF'] = RF_Reg.predict(X_train_OCF)
    RF_result = RF_Reg.predict(X_test_OCF)
    df_test['RF'] = RF_result.round(3)

    # OCF&DT
    bestParaDT = TuningParameter.tuningDT(X_train_OCF, y_train_OCF)
    DT_Reg = DecisionTreeRegressor(bestParaDT)
    DT_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['DT'] = DT_Reg.predict(X_train_OCF)
    DT_result = DT_Reg.predict(X_test_OCF)
    df_test['DT'] = DT_result.round(3)

    # OCF&KNN
    bestParaKNN = TuningParameter.tuningKNN(X_train_OCF, y_train_OCF)
    KNN_Reg = KNeighborsRegressor(bestParaKNN)
    KNN_Reg.fit(X_train_OCF, y_train_OCF)
    df_train['KNN'] = KNN_Reg.predict(X_train_OCF)
    X_test_KNN = preprocessing.minmax_scale(X_test_OCF)
    KNN_result = KNN_Reg.predict(X_test_KNN)
    df_test['KNN'] = KNN_result.round(3)

    # Voting Ensemble - VOCF
    voting_estimators = []
    voting_estimators.append(('MLR', MLR_Reg))
    voting_estimators.append(('SVR', SVR_Reg))
    voting_estimators.append(('MLP', MLP_Reg))
    voting_estimators.append(('GB', GB_Reg))
    voting_estimators.append(('RF', RF_Reg))
    voting_estimators.append(('DT', DT_Reg))
    voting_estimators.append(('KNN', KNN_Reg))

    voting = VotingRegressor(estimators=voting_estimators)
    voting.fit(X_train_OCF, y_train_OCF)
    df_train['VOCF'] = voting.predict(X_train_OCF)
    df_test['VOCF'] = voting.predict(X_test_OCF)

    # Stacking Ensemble - SOCF
    stacking_estimators = []
    stacking_estimators.append(('MLR', MLR_Reg))
    stacking_estimators.append(('SVR', SVR_Reg))
    stacking_estimators.append(('MLP', MLP_Reg))
    stacking_estimators.append(('GB', GB_Reg))
    stacking_estimators.append(('RF', RF_Reg))
    stacking_estimators.append(('DT', DT_Reg))
    stacking_estimators.append(('KNN', KNN_Reg))

    stacking = StackingRegressor(estimators=stacking_estimators, final_estimator=RF_Reg)
    stacking.fit(X_train_OCF, y_train_OCF)
    df_train['SOCF'] = stacking.predict(X_train_OCF)
    df_test['SOCF'] = stacking.predict(X_test_OCF)

    # --------
    # Evaluation
    df_test['SSEiOCF'] = pow(df_test['Real_P20'].astype(float) - df_test['OCF'].astype(float), 2)
    df_test['SSEiSVR'] = pow(df_test['Real_P20'].astype(float) - df_test['SVR'].astype(float), 2)
    df_test['SSEiMLP'] = pow(df_test['Real_P20'].astype(float) - df_test['MLP'].astype(float), 2)
    df_test['SSEiDT'] = pow(df_test['Real_P20'].astype(float) - df_test['DT'].astype(float), 2)
    df_test['SSEiMLR'] = pow(df_test['Real_P20'].astype(float) - df_test['MLR'].astype(float), 2)
    df_test['SSEiGB'] = pow(df_test['Real_P20'].astype(float) - df_test['GB'].astype(float), 2)
    df_test['SSEiRF'] = pow(df_test['Real_P20'].astype(float) - df_test['RF'].astype(float), 2)
    df_test['SSEiKNN'] = pow(df_test['Real_P20'].astype(float) - df_test['KNN'].astype(float), 2)
    df_test['SSEiVOCF'] = pow(df_test['Real_P20'].astype(float) - df_test['VOCF'].astype(float), 2)
    df_test['SSEiSOCF'] = pow(df_test['Real_P20'].astype(float) - df_test['SOCF'].astype(float), 2)

    df_test['MREiOCF'] = abs(df_test['OCF'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiSVR'] = abs(df_test['SVR'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiMLP'] = abs(df_test['MLP'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiDT'] = abs(df_test['DT'].astype(float) - df_test['Real_P20'].astype(float)) \
                          / df_test['Real_P20'].astype(float)
    df_test['MREiMLR'] = abs(df_test['MLR'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiGB'] = abs(df_test['GB'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiRF'] = abs(df_test['RF'].astype(float) - df_test['Real_P20'].astype(float)) \
                        / df_test['Real_P20'].astype(float)
    df_test['MREiKNN'] = abs(df_test['KNN'].astype(float) - df_test['Real_P20'].astype(float)) \
                        / df_test['Real_P20'].astype(float)
    df_test['MREiVOCF'] = abs(df_test['VOCF'].astype(float) - df_test['Real_P20'].astype(float)) \
                          / df_test['Real_P20'].astype(float)
    df_test['MREiSOCF'] = abs(df_test['SOCF'].astype(float) - df_test['Real_P20'].astype(float)) \
                          / df_test['Real_P20'].astype(float)

    SSE_OCF = df_test['SSEiOCF'].sum()
    SSE_SVR = df_test['SSEiSVR'].sum()
    SSE_MLP = df_test['SSEiMLP'].sum()
    SSE_DT = df_test['SSEiDT'].sum()
    SSE_MLR = df_test['SSEiMLR'].sum()
    SSE_GB = df_test['SSEiGB'].sum()
    SSE_RF = df_test['SSEiRF'].sum()
    SSE_KNN = df_test['SSEiKNN'].sum()
    SSE_VOCF = df_test['SSEiVOCF'].sum()
    SSE_SOCF = df_test['SSEiSOCF'].sum()

    predOCF_count = df_test.loc[df_test['MREiOCF'] < 0.25, 'MREiOCF'].count()
    Pred_OCF = predOCF_count / df_test['MREiOCF'].count()

    predSVR_count = df_test.loc[df_test['MREiSVR'] < 0.25, 'MREiSVR'].count()
    Pred_SVR = predSVR_count / df_test['MREiSVR'].count()

    predMLP_count = df_test.loc[df_test['MREiMLP'] < 0.25, 'MREiMLP'].count()
    Pred_MLP = predMLP_count / df_test['MREiMLP'].count()

    predDT_count = df_test.loc[df_test['MREiDT'] < 0.25, 'MREiDT'].count()
    Pred_DT = predDT_count / df_test['MREiDT'].count()

    predMLR_count = df_test.loc[df_test['MREiMLR'] < 0.25, 'MREiMLR'].count()
    Pred_MLR = predMLR_count / df_test['MREiMLR'].count()

    predGB_count = df_test.loc[df_test['MREiGB'] < 0.25, 'MREiGB'].count()
    Pred_GB = predGB_count / df_test['MREiGB'].count()

    predRF_count = df_test.loc[df_test['MREiRF'] < 0.25, 'MREiRF'].count()
    Pred_RF = predRF_count / df_test['MREiRF'].count()

    predKNN_count = df_test.loc[df_test['MREiKNN'] < 0.25, 'MREiKNN'].count()
    Pred_KNN = predKNN_count / df_test['MREiKNN'].count()

    predVOCF_count = df_test.loc[df_test['MREiVOCF'] < 0.25, 'MREiVOCF'].count()
    Pred_VOCF = predVOCF_count / df_test['MREiVOCF'].count()

    predSOCF_count = df_test.loc[df_test['MREiSOCF'] < 0.25, 'MREiSOCF'].count()
    Pred_SOCF = predSOCF_count / df_test['MREiSOCF'].count()

    MAE_OCF = mae(df_test['Real_P20'], df_test['OCF'])
    MAE_SVR = mae(df_test['Real_P20'], df_test['SVR'])
    MAE_MLP = mae(df_test['Real_P20'], df_test['MLP'])
    MAE_DT = mae(df_test['Real_P20'], df_test['DT'])
    MAE_MLR = mae(df_test['Real_P20'], df_test['MLR'])
    MAE_GB = mae(df_test['Real_P20'], df_test['GB'])
    MAE_RF = mae(df_test['Real_P20'], df_test['RF'])
    MAE_KNN = mae(df_test['Real_P20'], df_test['KNN'])
    MAE_VOCF = mae(df_test['Real_P20'], df_test['VOCF'])
    MAE_SOCF = mae(df_test['Real_P20'], df_test['SOCF'])

    RMSE_OCF = sqrt(SSE_OCF / df_test['SSEiOCF'].count())
    RMSE_SVR = sqrt(SSE_SVR / df_test['SSEiSVR'].count())
    RMSE_MLP = sqrt(SSE_MLP / df_test['SSEiMLP'].count())
    RMSE_DT = sqrt(SSE_DT / df_test['SSEiDT'].count())
    RMSE_MLR = sqrt(SSE_MLR / df_test['SSEiMLR'].count())
    RMSE_GB = sqrt(SSE_GB / df_test['SSEiGB'].count())
    RMSE_RF = sqrt(SSE_RF / df_test['SSEiRF'].count())
    RMSE_KNN = sqrt(SSE_KNN / df_test['SSEiKNN'].count())
    RMSE_VOCF = sqrt(SSE_VOCF / df_test['SSEiVOCF'].count())
    RMSE_SOCF = sqrt(SSE_SOCF / df_test['SSEiSOCF'].count())

    MdMRE_OCF = np.median(df_test['MREiOCF'])
    MdMRE_SVR = np.median(df_test['MREiSVR'])
    MdMRE_MLP = np.median(df_test['MREiMLP'])
    MdMRE_DT = np.median(df_test['MREiDT'])
    MdMRE_MLR = np.median(df_test['MREiMLR'])
    MdMRE_GB = np.median(df_test['MREiGB'])
    MdMRE_RF = np.median(df_test['MREiRF'])
    MdMRE_KNN = np.median(df_test['MREiKNN'])
    MdMRE_VOCF = np.median(df_test['MREiVOCF'])
    MdMRE_SOCF = np.median(df_test['MREiSOCF'])

    df_test['AEiOCF_min'] = abs(df_test['OCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['OCF', 'Real_P20']].min(axis=1)
    df_test['AEiAOM_min'] = abs(df_test['AOM'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['AOM', 'Real_P20']].min(axis=1)
    df_test['AEiSVR_min'] = abs(df_test['SVR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['SVR', 'Real_P20']].min(axis=1)
    df_test['AEiMLP_min'] = abs(df_test['MLP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['MLP', 'Real_P20']].min(axis=1)
    df_test['AEiDT_min'] = abs(df_test['DT'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['DT', 'Real_P20']].min(axis=1)
    df_test['AEiMLR_min'] = abs(df_test['MLR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                              df_test[['MLR', 'Real_P20']].min(axis=1)
    df_test['AEiGB_min'] = abs(df_test['GB'].astype(float) - df_test['Real_P20'].astype(float)) / \
                              df_test[['GB', 'Real_P20']].min(axis=1)
    df_test['AEiRF_min'] = abs(df_test['RF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                              df_test[['RF', 'Real_P20']].min(axis=1)
    df_test['AEiKNN_min'] = abs(df_test['KNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                           df_test[['KNN', 'Real_P20']].min(axis=1)
    df_test['AEiVOCF_min'] = abs(df_test['VOCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['VOCF', 'Real_P20']].min(axis=1)
    df_test['AEiSOCF_min'] = abs(df_test['SOCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['SOCF', 'Real_P20']].min(axis=1)

    df_test['AEiOCF_man'] = abs(df_test['OCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['OCF', 'Real_P20']].max(axis=1)
    df_test['AEiSVR_man'] = abs(df_test['SVR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['SVR', 'Real_P20']].max(axis=1)
    df_test['AEiMLP_man'] = abs(df_test['MLP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['MLP', 'Real_P20']].max(axis=1)
    df_test['AEiDT_man'] = abs(df_test['DT'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['DT', 'Real_P20']].max(axis=1)
    df_test['AEiMLR_man'] = abs(df_test['MLR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['MLR', 'Real_P20']].max(axis=1)
    df_test['AEiGB_man'] = abs(df_test['GB'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['GB', 'Real_P20']].max(axis=1)
    df_test['AEiRF_man'] = abs(df_test['RF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['RF', 'Real_P20']].max(axis=1)
    df_test['AEiKNN_man'] = abs(df_test['KNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                           df_test[['KNN', 'Real_P20']].max(axis=1)
    df_test['AEiVOCF_man'] = abs(df_test['VOCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                              df_test[['VOCF', 'Real_P20']].max(axis=1)
    df_test['AEiSOCF_man'] = abs(df_test['VOCF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['SOCF', 'Real_P20']].max(axis=1)

    MBRE_UCP = df_test['AEiUCP_min'].sum() / df_test['AEiUCP_min'].count()
    MBRE_AOM = df_test['AEiAOM_min'].sum() / df_test['AEiAOM_min'].count()
    MBRE_SVR = df_test['AEiSVR_min'].sum() / df_test['AEiSVR_min'].count()
    MBRE_MLP = df_test['AEiMLP_min'].sum() / df_test['AEiMLP_min'].count()
    MBRE_DT = df_test['AEiDT_min'].sum() / df_test['AEiDT_min'].count()
    MBRE_MLR = df_test['AEiMLR_min'].sum() / df_test['AEiMLR_min'].count()
    MBRE_GB = df_test['AEiGB_min'].sum() / df_test['AEiGB_min'].count()
    MBRE_RF = df_test['AEiRF_min'].sum() / df_test['AEiRF_min'].count()
    MBRE_KNN = df_test['AEiKNN_min'].sum() / df_test['AEiKNN_min'].count()
    MBRE_VOCF = df_test['AEiVOCF_min'].sum() / df_test['AEiVOCF_min'].count()
    MBRE_SOCF = df_test['AEiSOCF_min'].sum() / df_test['AEiSOCF_min'].count()

    MIBRE_OCF = df_test['AEiOCF_man'].sum() / df_test['AEiOCF_man'].count()
    MIBRE_SVR = df_test['AEiSVR_man'].sum() / df_test['AEiSVR_man'].count()
    MIBRE_MLP = df_test['AEiMLP_man'].sum() / df_test['AEiMLP_man'].count()
    MIBRE_DT = df_test['AEiDT_man'].sum() / df_test['AEiDT_man'].count()
    MIBRE_MLR = df_test['AEiMLR_man'].sum() / df_test['AEiMLR_man'].count()
    MIBRE_GB = df_test['AEiGB_man'].sum() / df_test['AEiGB_man'].count()
    MIBRE_RF = df_test['AEiRF_man'].sum() / df_test['AEiRF_man'].count()
    MIBRE_KNN = df_test['AEiKNN_man'].sum() / df_test['AEiKNN_man'].count()
    MIBRE_VOCF = df_test['AEiVOCF_man'].sum() / df_test['AEiVOCF_man'].count()
    MIBRE_SOCF = df_test['AEiSOCF_man'].sum() / df_test['AEiSOCF_man'].count()

    # -----
    result = [SSE_OCF.round(3), SSE_SVR.round(3),
              SSE_MLP.round(3), SSE_DT.round(3), SSE_MLR.round(3), SSE_GB.round(3), SSE_RF.round(3), SSE_KNN.round(3),
              SSE_VOCF.round(3), SSE_SOCF.round(3),

              Pred_OCF.round(3), Pred_SVR.round(3),
              Pred_MLP.round(3), Pred_DT.round(3), Pred_MLR.round(3), Pred_GB.round(3), Pred_RF.round(3), Pred_KNN.round(3),
              Pred_VOCF.round(3), Pred_SOCF.round(3),

              MAE_OCF.round(3), MAE_SVR.round(3),
              MAE_MLP.round(3), MAE_DT.round(3), MAE_MLR.round(3), MAE_GB.round(3), MAE_RF.round(3), MAE_KNN.round(3),
              MAE_VOCF.round(3), MAE_SOCF.round(3),

              MdMRE_OCF.round(3), MdMRE_SVR.round(3),
              MdMRE_MLP.round(3), MdMRE_DT.round(3), MdMRE_MLR.round(3), MdMRE_GB.round(3), MdMRE_RF.round(3), MdMRE_KNN.round(3),
              MdMRE_VOCF.round(3), MdMRE_SOCF.round(3),

              MBRE_UCP.round(3), MBRE_AOM.round(3), MBRE_SVR.round(3),
              MBRE_MLP.round(3), MBRE_DT.round(3), MBRE_MLR.round(3), MBRE_GB.round(3), MBRE_RF.round(3), MBRE_KNN.round(3),
              MBRE_VOCF.round(3), MBRE_SOCF.round(3),

              MIBRE_OCF.round(3), MIBRE_SVR.round(3),
              MIBRE_MLP.round(3), MIBRE_DT.round(3), MIBRE_MLR.round(3), MIBRE_GB.round(3), MIBRE_RF.round(3), MIBRE_KNN.round(3),
              MIBRE_VOCF.round(3), MIBRE_SOCF.round(3),

              RMSE_OCF, RMSE_SVR,
              RMSE_MLP, RMSE_DT, RMSE_MLR, RMSE_GB, RMSE_RF, RMSE_KNN,
              RMSE_VOCF, RMSE_SOCF]

    save_dataset(basedir, df_test, result)
    return result


def save_dataset(basedir, df_test, result):
    cols = df_test.T.__len__()
    last_row = []
    for i in range(cols - len(result)):
        last_row.append('-')
    for i in range(0, len(result)):
        last_row.append(result[i])

    df = pd.DataFrame([last_row], columns=df_test.columns)
    dfR = df_test.append(df, ignore_index=True)

    p = os.path.abspath('..')
    fn_r = os.path.join(p + '/datasetA' + "/" + basedir, 'result.csv')

    dfR.to_csv(fn_r)



