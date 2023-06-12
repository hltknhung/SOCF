# SOCF
# Incorporating Statistical and Machine Learning Techniques into the Optimization of Correction Factors for Software Development Effort Estimation

Abstract. 
Context: Accurate effort estimation is necessary for efficient management of software development projects, as it relates to human resource management. Ensemble methods, which employ multiple statistical and machine learning techniques, are more robust, reliable, and accurate effort estimation techniques. 
Objective: This study develops a stacking ensemble model based on Optimization Correction Factors by integrating seven statistical and machine learning techniques (K-Nearest Neighbour, Random Forest, Support Vector Regression, Multilayer Perception, Gradient Boosting, Linear Regression, and Decision Tree). The grid search optimization method is used to obtain valid search ranges and optimal configuration values, allowing more accurate estimation. Method: We conducted experiments to compare the proposed method with related methods, such as Use Case Points-based single methods, Optimization Correction Factors-based single methods, and ensemble methods. The estimation accuracies of the methods were evaluated using statistical tests and unbiased performance measures on a total of four datasets, thus demonstrating the effectiveness of the proposed method more clearly. 
Results: The proposed method successfully maintained its estimation accuracy across the four experimental datasets and gave the best results in terms of the SSE, MAE, RMSE, MBRE, MIBRE, MdMRE, and PRED (0.25). The p-value for the t-test showed that the proposed method is statistically superior to other methods in terms of estimation accuracy. 
Conclusion: The results show that the proposed method is a comprehensive approach for improving estimation accuracy and minimizing project risks in the early stages of software development.

# Dataset
A total of 70 projects from three repositories were used. Figure 6 shows boxplots of Real_P20 for each data repository, where Real_P20 is real effort in person-hours divided by productivity (PF - person-hours per 1 UCP). The repositories have significantly different Real_P20 values. Specifically, the D1 data repository has the largest Real_P20 for projects, while the D3 data repository has the smallest Real_P20 for projects. The D4 data repository, which combines D1 – D3, was used to evaluate the impact of mixing projects from different data repositories.

<img width="411" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/3325f6a6-4976-483d-8cd9-4e5e7cf3c910">

# Results
The ratio of improvement for which SOCF outperforms each other methods in terms of SSE, MAE, MBRE, MIBRE, MdMRE, and RMSE

<img width="253" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/3a5f4bee-56da-4fb5-b570-9ce17bb0f1cb">

<img width="251" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/2f7253ff-aa71-4f9a-a562-f5367454fcfe">

<img width="249" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/365f7e99-1d7b-4699-b3da-06394d2e2f74">

<img width="251" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/56c25cd7-da80-4a47-8587-3eed57d55432">

<img width="249" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/7fda2939-1092-432f-8eb2-2bac5658b5fb">

<img width="249" alt="image" src="https://github.com/hltknhung/JSME-22-0297/assets/58749238/e8fb7c64-2238-4cfd-8b95-00f8fd4c4569">



