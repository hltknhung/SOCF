import pandas as pd
from math import sqrt

import sklearn.ensemble
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import os
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from pyGRNN import GRNN
import TuningParameter
import time

def run_cases(df_train, df_test, basedir):
    df_train['UCP'] = (df_train['UAW'].astype(float) + df_train['UUCW'].astype(float)) \
                      * df_train['TCF'].astype(float) * df_train['ECF'].astype(float)
    df_test['UCP'] = (df_test['UAW'].astype(float) + df_test['UUCW'].astype(float)) \
                     * df_test['TCF'].astype(float) * df_test['ECF'].astype(float)

    df_train['x1_UCP'] = df_train['UAW'].astype(float) * df_train['TCF'].astype(float) * df_train['ECF'].astype(
        float)
    df_train['x2_UCP'] = df_train['UUCW'].astype(float) * df_train['TCF'].astype(float) * df_train['ECF'].astype(
        float)
    df_test['x1_UCP'] = df_test['UAW'].astype(float) * df_test['TCF'].astype(float) * df_test['ECF'].astype(float)
    df_test['x2_UCP'] = df_test['UUCW'].astype(float) * df_test['TCF'].astype(float) * df_test['ECF'].astype(float)

    X_train_UCP = df_train[['x1_UCP', 'x2_UCP']]
    X_train_UCP = preprocessing.minmax_scale(X_train_UCP)
    y_train_UCP = df_train['Real_P20']
    X_test_UCP = df_test[['x1_UCP_MLP', 'x2_UCP_MLP']]
    X_test_UCP = preprocessing.minmax_scale(X_test_UCP)

    # MLP
    bestParaMLP = TuningParameter.tuningMLP(X_train_UCP, y_train_UCP)
    UCP_MLP_Reg = MLPRegressor(bestParaMLP)
    UCP_MLP_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_MLP'] = UCP_MLP_Reg.predict(X_train_UCP)
    UCP_MLP_result = UCP_MLP_Reg.predict(X_test_UCP)
    df_test['UCP_MLP'] = UCP_MLP_result.round(3)

    # GRNN
    UCP_GRNN_Reg = GRNN(method='nelder-mead', sigma=0.1)
    UCP_GRNN_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_GRNN'] = UCP_GRNN_Reg.predict(X_train_UCP)
    UCP_GRNN_result = UCP_GRNN_Reg.predict(X_test_UCP)
    df_test['UCP_GRNN'] = UCP_GRNN_result.round(3)

    # RF
    bestParaRF = TuningParameter.tuningRF(X_train_UCP, y_train_UCP)
    UCP_RF_Reg = RandomForestRegressor(bestParaRF)
    UCP_RF_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_RF'] = UCP_RF_Reg.predict(X_train_UCP)
    UCP_RF_result = UCP_RF_Reg.predict(X_test_UCP)
    df_test['UCP_RF'] = UCP_RF_result.round(3)

    # DT
    bestParaDT = TuningParameter.tuningDT(X_train_UCP, y_train_UCP)
    UCP_DT_Reg = DecisionTreeRegressor(bestParaDT)
    UCP_DT_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_DT'] = UCP_DT_Reg.predict(X_train_UCP)
    UCP_DT_result = UCP_DT_Reg.predict(X_test_UCP)
    df_test['UCP_DT'] = UCP_DT_result.round(3)

    # SVR
    bestParaSVR = TuningParameter.tuningSVR(X_train_UCP, y_train_UCP)
    UCP_SVR_Reg = SVR(bestParaSVR)
    UCP_SVR_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_SVR'] = UCP_SVR_Reg.predict(X_train_UCP)
    UCP_SVR_result = UCP_SVR_Reg.predict(X_test_UCP)
    df_test['UCP_SVR'] = UCP_SVR_result.round(3)

    # KNN
    bestParaKNN = TuningParameter.tuningKNN(X_train_UCP, y_train_UCP)
    UCP_KNN_Reg = KNeighborsRegressor(bestParaKNN)
    UCP_KNN_Reg.fit(X_train_UCP, y_train_UCP)
    df_train['UCP_KNN'] = UCP_KNN_Reg.predict(X_train_UCP)
    UCP_KNN_result = UCP_KNN_Reg.predict(X_test_UCP)
    df_test['UCP_KNN'] = UCP_KNN_result.round(3)

    # Adaboost Ensemble - UCP
    UCPada = AdaBoostRegressor(base_estimator=UCP_KNN_Reg)
    UCPada.fit(X_train_UCP, y_train_UCP)
    UCPada = AdaBoostRegressor(base_estimator=UCP_SVR_Reg)
    UCPada.fit(X_train_UCP, y_train_UCP)
    df_train['AUCP'] = UCPada.predict(X_train_UCP)
    df_test['AUCP'] = UCPada.predict(X_test_UCP)

    # Voting Ensemble - UCP
    UCPvoting_estimators = []
    UCPvoting_estimators.append(('MLR', UCP_KNN_Reg))
    UCPvoting_estimators.append(('SVR', UCP_SVR_Reg))
    UCPvoting_estimators.append(('MLP', UCP_DT_Reg))
    UCPvoting = VotingRegressor(estimators=UCPvoting_estimators)
    UCPvoting.fit(X_train_UCP, y_train_UCP)
    df_train['VUCP'] = UCPvoting.predict(X_train_UCP)
    df_test['VUCP'] = UCPvoting.predict(X_test_UCP)

    # --------
    # Evaluation
    df_test['SSEiUCP'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP'].astype(float), 2)
    df_test['SSEiUCP_SVR'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_SVR'].astype(float), 2)
    df_test['SSEiUCP_MLP'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_MLP'].astype(float), 2)
    df_test['SSEiUCP_GRNN'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_GRNN'].astype(float), 2)
    df_test['SSEiUCP_KNN'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_KNN'].astype(float), 2)
    df_test['SSEiUCP_DT'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_DT'].astype(float), 2)
    df_test['SSEiUCP_RF'] = pow(df_test['Real_P20'].astype(float) - df_test['UCP_RF'].astype(float), 2)
    df_test['SSEiVUCP'] = pow(df_test['Real_P20'].astype(float) - df_test['VUCP'].astype(float), 2)
    df_test['SSEiAUCP'] = pow(df_test['Real_P20'].astype(float) - df_test['AUCP'].astype(float), 2)

    df_test['MREiUCP'] = abs(df_test['UCP'].astype(float) - df_test['Real_P20'].astype(float)) \
                         / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_SVR'] = abs(df_test['UCP_SVR'].astype(float) - df_test['Real_P20'].astype(float)) \
                             / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_MLP'] = abs(df_test['UCP_MLP'].astype(float) - df_test['Real_P20'].astype(float)) \
                             / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_GRNN'] = abs(df_test['UCP_GRNN'].astype(float) - df_test['Real_P20'].astype(float)) \
                              / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_KNN'] = abs(df_test['UCP_KNN'].astype(float) - df_test['Real_P20'].astype(float)) \
                             / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_DT'] = abs(df_test['UCP_DT'].astype(float) - df_test['Real_P20'].astype(float)) \
                            / df_test['Real_P20'].astype(float)
    df_test['MREiUCP_RF'] = abs(df_test['UCP_RF'].astype(float) - df_test['Real_P20'].astype(float)) \
                            / df_test['Real_P20'].astype(float)
    df_test['MREiVUCP'] = abs(df_test['VUCP'].astype(float) - df_test['Real_P20'].astype(float)) \
                          / df_test['Real_P20'].astype(float)
    # df_test['MREiBUCP'] = abs(df_test['BUCP'].astype(float) - df_test['Real_P20'].astype(float)) \
    #                      / df_test['Real_P20'].astype(float)
    df_test['MREiAUCP'] = abs(df_test['AUCP'].astype(float) - df_test['Real_P20'].astype(float)) \
                          / df_test['Real_P20'].astype(float)

    SSE_UCP = df_test['SSEiUCP'].sum()
    SSE_UCP_SVR = df_test['SSEiUCP_SVR'].sum()
    SSE_UCP_MLP = df_test['SSEiUCP_MLP'].sum()
    SSE_UCP_GRNN = df_test['SSEiUCP_GRNN'].sum()
    SSE_UCP_KNN = df_test['SSEiUCP_KNN'].sum()
    SSE_UCP_DT = df_test['SSEiUCP_DT'].sum()
    SSE_UCP_RF = df_test['SSEiUCP_RF'].sum()
    SSE_VUCP = df_test['SSEiVUCP'].sum()
    SSE_AUCP = df_test['SSEiAUCP'].sum()

    predUCP_count = df_test.loc[df_test['MREiUCP'] < 0.25, 'MREiUCP'].count()
    Pred_UCP = predUCP_count / df_test['MREiUCP'].count()

    predUCP_SVR_count = df_test.loc[df_test['MREiUCP_SVR'] < 0.25, 'MREiUCP_SVR'].count()
    Pred_UCP_SVR = predUCP_SVR_count / df_test['MREiUCP_SVR'].count()

    predUCP_MLP_count = df_test.loc[df_test['MREiUCP_MLP'] < 0.25, 'MREiUCP_MLP'].count()
    Pred_UCP_MLP = predUCP_MLP_count / df_test['MREiUCP_MLP'].count()

    predUCP_GRNN_count = df_test.loc[df_test['MREiUCP_GRNN'] < 0.25, 'MREiUCP_GRNN'].count()
    Pred_UCP_GRNN = predUCP_GRNN_count / df_test['MREiUCP_GRNN'].count()

    predUCP_KNN_count = df_test.loc[df_test['MREiUCP_KNN'] < 0.25, 'MREiUCP_KNN'].count()
    Pred_UCP_KNN = predUCP_KNN_count / df_test['MREiUCP_KNN'].count()

    predUCP_DT_count = df_test.loc[df_test['MREiUCP_DT'] < 0.25, 'MREiUCP_DT'].count()
    Pred_UCP_DT = predUCP_DT_count / df_test['MREiUCP_DT'].count()

    predUCP_RF_count = df_test.loc[df_test['MREiUCP_RF'] < 0.25, 'MREiUCP_RF'].count()
    Pred_UCP_RF = predUCP_RF_count / df_test['MREiUCP_RF'].count()

    predVUCP_count = df_test.loc[df_test['MREiVUCP'] < 0.25, 'MREiVUCP'].count()
    Pred_VUCP = predVUCP_count / df_test['MREiVUCP'].count()

    predAUCP_count = df_test.loc[df_test['MREiAUCP'] < 0.25, 'MREiAUCP'].count()
    Pred_AUCP = predAUCP_count / df_test['MREiAUCP'].count()

    MAE_UCP = mae(df_test['Real_P20'], df_test['UCP'])
    MAE_UCP_SVR = mae(df_test['Real_P20'], df_test['UCP_SVR'])
    MAE_UCP_MLP = mae(df_test['Real_P20'], df_test['UCP_MLP'])
    MAE_UCP_GRNN = mae(df_test['Real_P20'], df_test['UCP_GRNN'])
    MAE_UCP_KNN = mae(df_test['Real_P20'], df_test['UCP_KNN'])
    MAE_UCP_DT = mae(df_test['Real_P20'], df_test['UCP_DT'])
    MAE_UCP_RF = mae(df_test['Real_P20'], df_test['UCP_RF'])
    MAE_VUCP = mae(df_test['Real_P20'], df_test['VUCP'])
    MAE_AUCP = mae(df_test['Real_P20'], df_test['AUCP'])

    RMSE_UCP = sqrt(SSE_UCP / df_test['SSEiUCP'].count())
    RMSE_UCP_SVR = sqrt(SSE_UCP_SVR / df_test['SSEiUCP_SVR'].count())
    RMSE_UCP_MLP = sqrt(SSE_UCP_MLP / df_test['SSEiUCP_MLP'].count())
    RMSE_UCP_GRNN = sqrt(SSE_UCP_GRNN / df_test['SSEiUCP_GRNN'].count())
    RMSE_UCP_KNN = sqrt(SSE_UCP_KNN / df_test['SSEiUCP_KNN'].count())
    RMSE_UCP_DT = sqrt(SSE_UCP_DT / df_test['SSEiUCP_DT'].count())
    RMSE_UCP_RF = sqrt(SSE_UCP_RF / df_test['SSEiUCP_RF'].count())
    RMSE_VUCP = sqrt(SSE_VUCP / df_test['SSEiVUCP'].count())
    RMSE_AUCP = sqrt(SSE_AUCP / df_test['SSEiAUCP'].count())

    MdMRE_UCP = np.median(df_test['MREiUCP'])
    MdMRE_UCP_SVR = np.median(df_test['MREiUCP_SVR'])
    MdMRE_UCP_MLP = np.median(df_test['MREiUCP_MLP'])
    MdMRE_UCP_GRNN = np.median(df_test['MREiUCP_GRNN'])
    MdMRE_UCP_KNN = np.median(df_test['MREiUCP_KNN'])
    MdMRE_UCP_DT = np.median(df_test['MREiUCP_DT'])
    MdMRE_UCP_RF = np.median(df_test['MREiUCP_RF'])
    MdMRE_VUCP = np.median(df_test['MREiVUCP'])
    MdMRE_AUCP = np.median(df_test['MREiAUCP'])

    df_test['AEiUCP_min'] = abs(df_test['UCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['UCP', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_SVR_min'] = abs(df_test['UCP_SVR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_SVR', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_MLP_min'] = abs(df_test['UCP_MLP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_MLP', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_GRNN_min'] = abs(df_test['UCP_GRNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                 df_test[['UCP_GRNN', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_KNN_min'] = abs(df_test['UCP_KNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_KNN', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_DT_min'] = abs(df_test['UCP_DT'].astype(float) - df_test['Real_P20'].astype(float)) / \
                               df_test[['UCP_DT', 'Real_P20']].min(axis=1)
    df_test['AEiUCP_RF_min'] = abs(df_test['UCP_RF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                               df_test[['UCP_RF', 'Real_P20']].min(axis=1)
    df_test['AEiVUCP_min'] = abs(df_test['VUCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['VUCP', 'Real_P20']].min(axis=1)
    df_test['AEiAUCP_min'] = abs(df_test['AUCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['AUCP', 'Real_P20']].min(axis=1)

    df_test['AEiUCP_man'] = abs(df_test['UCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                            df_test[['UCP', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_SVR_man'] = abs(df_test['UCP_SVR'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_SVR', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_MLP_man'] = abs(df_test['UCP_MLP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_MLP', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_GRNN_man'] = abs(df_test['UCP_GRNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                 df_test[['UCP_GRNN', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_KNN_man'] = abs(df_test['UCP_KNN'].astype(float) - df_test['Real_P20'].astype(float)) / \
                                df_test[['UCP_KNN', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_DT_man'] = abs(df_test['UCP_DT'].astype(float) - df_test['Real_P20'].astype(float)) / \
                               df_test[['UCP_DT', 'Real_P20']].max(axis=1)
    df_test['AEiUCP_RF_man'] = abs(df_test['UCP_RF'].astype(float) - df_test['Real_P20'].astype(float)) / \
                               df_test[['UCP_RF', 'Real_P20']].max(axis=1)
    df_test['AEiVUCP_man'] = abs(df_test['VUCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['VUCP', 'Real_P20']].max(axis=1)
    # df_test['AEiBUCP_man'] = abs(df_test['BUCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
    #                            df_test[['BUCP', 'Real_P20']].max(axis=1)
    df_test['AEiAUCP_man'] = abs(df_test['AUCP'].astype(float) - df_test['Real_P20'].astype(float)) / \
                             df_test[['AUCP', 'Real_P20']].max(axis=1)

    MBRE_UCP = df_test['AEiUCP_min'].sum() / df_test['AEiUCP_min'].count()
    MBRE_UCP_SVR = df_test['AEiUCP_SVR_min'].sum() / df_test['AEiUCP_SVR_min'].count()
    MBRE_UCP_MLP = df_test['AEiUCP_MLP_min'].sum() / df_test['AEiUCP_MLP_min'].count()
    MBRE_UCP_GRNN = df_test['AEiUCP_GRNN_min'].sum() / df_test['AEiUCP_GRNN_min'].count()
    MBRE_UCP_KNN = df_test['AEiUCP_KNN_min'].sum() / df_test['AEiUCP_KNN_min'].count()
    MBRE_UCP_DT = df_test['AEiUCP_DT_min'].sum() / df_test['AEiUCP_DT_min'].count()
    MBRE_UCP_RF = df_test['AEiUCP_RF_min'].sum() / df_test['AEiUCP_RF_min'].count()
    MBRE_VUCP = df_test['AEiVUCP_min'].sum() / df_test['AEiVUCP_min'].count()
    MBRE_AUCP = df_test['AEiAUCP_min'].sum() / df_test['AEiAUCP_min'].count()

    MIBRE_UCP = df_test['AEiUCP_man'].sum() / df_test['AEiUCP_man'].count()
    MIBRE_UCP_SVR = df_test['AEiUCP_SVR_man'].sum() / df_test['AEiUCP_SVR_man'].count()
    MIBRE_UCP_MLP = df_test['AEiUCP_MLP_man'].sum() / df_test['AEiUCP_MLP_man'].count()
    MIBRE_UCP_GRNN = df_test['AEiUCP_GRNN_man'].sum() / df_test['AEiUCP_GRNN_man'].count()
    MIBRE_UCP_KNN = df_test['AEiUCP_KNN_man'].sum() / df_test['AEiUCP_KNN_man'].count()
    MIBRE_UCP_DT = df_test['AEiUCP_DT_man'].sum() / df_test['AEiUCP_DT_man'].count()
    MIBRE_UCP_RF = df_test['AEiUCP_RF_man'].sum() / df_test['AEiUCP_RF_man'].count()
    MIBRE_VUCP = df_test['AEiVUCP_man'].sum() / df_test['AEiVUCP_man'].count()
    # MIBRE_BUCP = df_test['AEiBUCP_man'].sum() / df_test['AEiBUCP_man'].count()
    MIBRE_AUCP = df_test['AEiAUCP_man'].sum() / df_test['AEiAUCP_man'].count()

    # -----
    result = [SSE_UCP.round(3), SSE_UCP_SVR.round(3), SSE_UCP_MLP.round(3), SSE_UCP_GRNN.round(3), SSE_UCP_KNN.round(3),
              SSE_UCP_DT.round(3), SSE_UCP_RF.round(3), SSE_VUCP.round(3), SSE_AUCP.round(3),

              Pred_UCP.round(3), Pred_UCP_SVR.round(3), Pred_UCP_MLP.round(3), Pred_UCP_GRNN.round(3), Pred_UCP_KNN.round(3),
              Pred_UCP_DT.round(3), Pred_UCP_RF.round(3), Pred_VUCP.round(3), Pred_AUCP.round(3),

              MAE_UCP.round(3), MAE_UCP_SVR.round(3), MAE_UCP_MLP.round(3), MAE_UCP_GRNN.round(3), MAE_UCP_KNN.round(3),
              MAE_UCP_DT.round(3), MAE_UCP_RF.round(3), MAE_VUCP.round(3), MAE_AUCP.round(3),

              MdMRE_UCP.round(3), MdMRE_UCP_SVR.round(3), MdMRE_UCP_MLP.round(3), MdMRE_UCP_GRNN.round(3), MdMRE_UCP_KNN.round(3),
              MdMRE_UCP_DT.round(3), MdMRE_UCP_RF.round(3), MdMRE_VUCP.round(3), MdMRE_AUCP.round(3),

              MBRE_UCP.round(3), MBRE_UCP_SVR.round(3), MBRE_UCP_MLP.round(3), MBRE_UCP_GRNN.round(3), MBRE_UCP_KNN.round(3),
              MBRE_UCP_DT.round(3), MBRE_UCP_RF.round(3), MBRE_VUCP.round(3), MBRE_AUCP.round(3),

              MIBRE_UCP.round(3), MIBRE_UCP_SVR.round(3), MIBRE_UCP_MLP.round(3), MIBRE_UCP_GRNN.round(3), MIBRE_UCP_KNN.round(3),
              MIBRE_UCP_DT.round(3), MIBRE_UCP_RF.round(3), MIBRE_VUCP.round(3), MIBRE_AUCP.round(3),

              RMSE_UCP, RMSE_UCP_SVR, RMSE_UCP_MLP, RMSE_UCP_GRNN, RMSE_UCP_KNN, RMSE_UCP_DT,
              RMSE_UCP_RF, RMSE_VUCP, RMSE_AUCP]

    save_dataset(basedir, df_test, result)
    return result

def save_dataset(basedir, df_test, result):
    cols = df_test.T.__len__()
    last_row = []
    for i in range(cols - len(result)):
        last_row.append('-')
    for i in range(0, len(result)):
        last_row.append(result[i])

    df = pd.DataFrame([last_row], columns = df_test.columns)
    dfR = df_test.append(df, ignore_index=True)

    p = os.path.abspath('..')
    fn_r = os.path.join(p + '/datasetA' + "/" + basedir, 'result.csv')

    dfR.to_csv(fn_r)

