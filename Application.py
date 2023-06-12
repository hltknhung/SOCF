import pandas as pd
from runAll import runAll

def xuly(rs, n):
    df = pd.DataFrame(rs)
    row = []
    for i in range(n):
        row.append(df[i].mean())
    return row

if __name__ == "__main__":

    columns = ["SSE_UCP",
              "SSE_UCP_SVR", "SSE_UCP_MLP", "SSE_UCP_GRNN", "SSE_UCP_KNN",
              "SSE_UCP_DT", "SSE_UCP_RF", "SSE_VUCP", "SSE_AUCP",
           "Pred_UCP",
              "Pred_UCP_SVR", "Pred_UCP_MLP", "Pred_UCP_GRNN", "Pred_UCP_KNN",
              "Pred_UCP_DT", "Pred_UCP_RF", "Pred_VUCP", "Pred_AUCP",
           "MAE_UCP",
              "MAE_UCP_SVR", "MAE_UCP_MLP", "MAE_UCP_GRNN", "MAE_UCP_KNN",
              "MAE_UCP_DT", "MAE_UCP_RF", "MAE_VUCP", "MAE_AUCP",
           "MdMRE_UCP",
              "MdMRE_UCP_SVR", "MdMRE_UCP_MLP", "MdMRE_UCP_GRNN", "MdMRE_UCP_KNN",
              "MdMRE_UCP_DT", "MdMRE_UCP_RF", "MdMRE_VUCP", "MdMRE_AUCP",
           "MBRE_UCP",
              "MBRE_UCP_SVR", "MBRE_UCP_MLP", "MBRE_UCP_GRNN", "MBRE_UCP_KNN",
              "MBRE_UCP_DT", "MBRE_UCP_RF", "MBRE_VUCP", "MBRE_AUCP",
           "MIBRE_UCP",
              "MIBRE_UCP_SVR", "MIBRE_UCP_MLP", "MIBRE_UCP_GRNN", "MIBRE_UCP_KNN",
              "MIBRE_UCP_DT", "MIBRE_UCP_RF", "MIBRE_VUCP", "MIBRE_AUCP",
           "RMSE_UCP",
           "RMSE_UCP_SVR", "RMSE_UCP_MLP", "RMSE_UCP_GRNN", "RMSE_UCP_KNN",
           "RMSE_UCP_DT", "RMSE_UCP_RF", "RMSE_VUCP", "RMSE_AUCP"]

    rs1 = runAll('HO1')
    rs2 = runAll('HO2')
    rs3 = runAll('HO3')
    rs4 = runAll('HO4')
    rs5 = runAll('HO5')
    # rs6 = runAll('HO6')
    # rs7 = runAll('HO7')
    # rs8 = runAll('HO8')
    # rs9 = runAll('HO9')
    # rs10 = runAll('HO10')

    n = len(columns)
    row1 = xuly(rs1, n)
    row2 = xuly(rs2, n)
    row3 = xuly(rs3, n)
    row4 = xuly(rs4, n)
    row5 = xuly(rs5, n)
    # row6 = xuly(rs6, n)
    # row7 = xuly(rs7, n)
    # row8 = xuly(rs8, n)
    # row9 = xuly(rs9, n)
    # row10 = xuly(rs10, n)

    arr = [row1, row2, row3, row4, row5]
    #arr = [row1, row5]
    df = pd.DataFrame(arr)

    print("Average of 5 runs:")
    for i in range(len(columns)):
        print(df[i].mean().round(3))
    df = pd.DataFrame(arr, columns=columns)

    print(df)
    # last_row = []
    # last_row.append(df.mean().round(3))
    # df = df.append(last_row, ignore_index=True)
    # df = df.rename(index={5: 'AVG of 5 runs'})

    df.to_csv("AverageResult.csv")
    print("DONE")