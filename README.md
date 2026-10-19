# 项目概述

### 项目背景

人口老龄化问题突出，我国养老保险“三支柱”体系发展失衡。商业养老保险作为“第三支柱”扮演重要角色，但目前发展水平远低于发达国家，发展潜力巨大。

国家高度重视数字技术与数据要素对金融业高质量发展的驱动作用。《关于做好金融“五篇大文章”的指导意见》《银行业保险业数字金融高质量发展实施方案》出台，为保险业数字化发展提供政策依据。

保险销售模式已从依赖人海战术的传统线下推销，深刻转向以互联网和数字化平台为核心的线上营销。在数字化时代，研究如何利用机器学习等先进数据分析技术，深入挖掘客户数据价值，精准预测和识别商业养老保险的购买意愿与行为，对于保险公司降本增效、优化产品设计、提升服务体验具有至关重要的现实意义。

### 研究意义

本项目立足于我国快速老龄化与养老保障体系亟待完善的宏观背景，紧扣国家推动数字金融发展的政策导向，针对保险业销售模式数字化转型中的核心痛点——精准客户识别，旨在通过机器学习技术分析我国居民商业养老保险购买行为，为保险机构实现精准营销、产品创新提供数据驱动的决策支持，同时为丰富养老第三支柱、积极应对人口老龄化贡献实践方案。

### 研究内容

基于中国综合社会调查数据，利用随机森林、XGBoost、LightGBM等集成学习算法对商业养老保险的购买决策进行分析，利用机器学习方法识别对个体购买商业养老保险意愿具有重要影响的因素，并探究其影响机制。

# 数据集

### 数据来源

本项目的数据来源于中国综合社会调查（CGSS）2021和2023年调查问卷的调查结果。该调查是中国最早的全国性、综合性、连续性学术调查项目，系统收集了社会、社区、家庭、个人多个层次的数据，覆盖全国31个省、自治区、直辖市（不含港澳台），调查对象为18-69岁居民。由于该调查不仅收录了受访者商业养老保险的参保情况，还覆盖了对商业养老保险购买行为具有潜在影响的大部分变量。并且该问卷是在全国范围内进行调查的，在样本选择上具有很强的随机性，一定程度上保证了分析结果的稳健性与代表性。

CGSS2023调查数据共有11326个样本，在对该数据集进行探索性分析时发现，由于2023年数据集对一些问题采取了随机抽样访问的搜集方法，因此本项目所需的一些解释变量存在比较严重缺失情况，例如：对儿子、女儿数量的回答分别有4360、4359个缺失值；对新媒体、传统媒体的使用情况分别有4347、4350个缺失值。对这些样本进行剔除会严重影响样本量，可能会使机器学习模型没有足够的样本去学习变量间的关系。对缺失值进行差值可能会模糊真实数据之间的联系，降低预测性能。因此，本项目将CGSS2023调查数据与CGSS2021调查数据合并进行研究，研究者认为合并后更大样本量带来的性能提升大于两期调查数据时间差别带来的负面影响。合并后数据集中共有19474个原始样本。

### 变量选取与特征工程

本项目共选取1个被解释变量与22个解释变量。解释变量为：年龄、婚姻状况、家庭规模、受教育情况、儿子数量、女儿数量、房产数量、代际经济支持、代际情感支持、代际生活照料、传统媒体使用频率、新型媒体使用频率、社会互动频率、居住地类型、家庭年总收入、幸福感、负面情绪、自评健康、是否有基本养老保险、是否有基本医疗保险、养老观念、金融素养。被解释变量为：是否购买了商业养老保险。

变量具体构造方法及变量说明见文件“1_1_变量说明”

变量描述性统计见文件“2_x”

# 方法论

### 不平衡样本处理
本研究所使用的被解释变量存在严重的不平衡问题。这会导致训练集中阳性样本数量严重不足，机器学习模型无法通过分析充足的阳性样本以捕捉变量间的隐含关系，无法达到理想的训练效果，因此本研究采用smote过采样技术对训练集中的阳性样本进行扩充。

对于测试集，阳性样本过少会使模型评价指标无法真实反映机器学习模型的预测效果，机器学习模型总可以通过预测阴性样本来获得较高的准确率，因此需要平衡测试集中两类样本的比例。由于smote过采样法会在数据中加入并不真实存在的合成样本，会给数据集产生噪声，掩盖变量间的真实联系，因此对于测试集数据，本研究采用随机剔除一定数量的阴性样本来平衡两类样本数量。

### 模型评价指标

对于二分类机器学习模型，常见的评价指标有混淆矩阵、精确率、召回率、ROC曲线及AUC值等。

对于本项目所研究的问题，由于阴、阳性样本处于高度不平衡，传统的评价指标正确率并不能很好的反映出模型的预测性能。考虑到在保险销售的实际应用中，更值得关注的是有意愿购买保险的个体，即阳性样本，因此对模型表现的评价应该侧重于在识别阳性样本的表现上。又因为对于保险销售来说，遗漏掉潜在客户的代价大于将保险推销给没有购买保险意愿的个体的代价，因此本研究将重点关注模型的阳性样本召回率表现，同时关注F1分数的表现情况，综合评价模型对阳性样本预测能力。

# 核心结果

### 模型对比
在三种机器学习模型的预测结果中，XGBoost模型拥有最高的召回率（XGBoost：0.72、随机森林：0.55、LightGBM：0.58），表示在所有商业养老保险的潜在客户中，该模型成功识别了其中的72%，其他模型的该比例显著低于XGBoost模型表现，而这在实际保险销售中非常重要。虽然XGBoost模型的精确率在三个模中最低（XGBoost：0.46、随机森林：0.52、LightGBM：0.52），但与随机森林、LightGBM模型的表现差距不大，表示在所有被预测为会购买商业养老保险的样本中，有46%的样本实际会购买。相比之下，拥有最较精确率的LightGBM模型的召回率只有0.58，意味着该模型会遗漏42%的潜在客户，这在实际应用中会为保险销售方造成巨大的收入损失。综合来看，XGBoost模型拥有最高的F1得分（0.56）和AUC值（0.76）。因此，本文选取XGBoost模型为最优模型进行进一步分析。¶（三种模型具体结果见文件“7_0”“7_1”“7_2”“7_3”）

### SHAP值分析
SHAP值分析显示，对居民是否购买商业养老保险具有重要预测能力的前十个特征为：受教育程度、是否居住在城市、是否具有基本医疗保险、是否具备金融素养、沮丧程度评分、家庭房产数量、新型媒体使用频率、女儿数量、传统媒体使用频率、年龄。（具体分析及结果展示见文件“8_1_SHAP值分析”）

### 偏依赖图分析
偏依赖图分析揭示了“受教育程度”“消极情绪”“新型/传统媒体使用情况”与商业养老保险购买概率的正相关关系，以及“女儿数量”“年龄”“房产数量”与购买概率之间的非线性关系。（具体分析及结果展示见文件“8_2_偏依赖图分析”）

# 主要结论

在利用机器学习模型对商业养老保险购买行为进行预测时，对非平衡样本的处理方法尤为重要。本研究经过反复尝试，发现通过smote过采样技术将训练集中正负样本比调整为1:1，并通过随机剔除多数样本将测试集中的正负样本比调整为3:7能让所有机器学习模型的整体性能最优，其中XGBoost模型为最优模型。

本研究的所有特征中，对商业养老保险购买行为具有显著预测能力的前十大特征为：受教育程度、是否居住在城市、是否具有基本医疗保险、是否具备金融素养、家庭房产数量、沮丧程度评分、女儿数量、新型媒体使用频率、传统媒体使用频率、年龄。目前的保险营销实务中对年龄、收入、房产数量、金融素养、居住地等特征关注度很高，但可能受限于信息可获得性等多种因素，实务中对受教育程度、沮丧程度评分、媒体使用频率等特征关注度较低。未来保险公司可以在合法合规的前提下着重收集客户的此类特征信息。

本研究发现，具有本科及以上学历、居住在城市、从事风险投资活动、家庭拥有2套及以上房产、较少感到负面情绪、拥有1个女儿、经常使用新型或传统媒体、年龄在55岁-65岁之间这些特征之一的个体购买商业养老保险的概率会明显高于不具备以上特征的个体。保险销售者在精准营销过程中可以重点关注以上特征，并且拓展保险广告的投放渠道，注重新型媒体对保险营销的重要作用。

# 局限性

由于数据集存在严重的不平衡性，本文在构建训练集时运用了过采样技术，这可能会模糊少数类样本与多数类样本之间的分界线，影响学习效果。同时由于数据集的不平衡问题，机器学习模型在预测过程中难免倾向于预测阴性样本，夸大特征对商业养老保险购买概率的负向影响，导致特征的shap值普遍呈现左偏。

受限于原始数据覆盖范围有限或存在较严重的缺失情况，本研究不得不剔除部分可能具有显著预测能力特征，如代际经济支持、代际情感支持、特定职业、特定家庭特征等。希望可以利用更加详尽的数据去尝试包含更多的特征组合，以挖掘更多影响因素，改善模型性能，为保险销售者提供更有价值的建议。

本研究的数据来源以及所选特征更多来源于学术性社会调查以及相关学术论文，所得结论更偏向于理论结果。对于实际业务中的预测效果，研究者希望可以利用保险公司所收集的真实数据库进行预测，来验证该模型在实务中的识别效果。

# 如何运行
按文件编号顺序依次运行所有代码文件即可得到全部研究结果，主要图表及结果解读均在代码文件中展示。




## 1. Project Overview

### 1.1 Background

China is experiencing rapid population aging, while its three-pillar pension system remains unevenly developed. As the third pillar, commercial pension insurance plays an important role in supplementing retirement security. However, its development remains relatively limited compared with that in many developed economies, leaving considerable room for growth.

The Chinese government has increasingly emphasized the role of digital technologies and data as key drivers of high-quality financial development. Policy initiatives, including the *Guiding Opinions on Advancing the Five Major Areas of Finance* and the *Implementation Plan for High-Quality Development of Digital Finance in the Banking and Insurance Sectors*, provide policy support for the digital transformation of the insurance industry.

Insurance distribution has also shifted from traditional offline sales models toward online marketing and digital platforms. In this context, machine learning and advanced data analytics can help insurers extract value from customer data and identify potential customers more effectively. Understanding the factors associated with commercial pension insurance purchases may help insurers improve marketing efficiency, optimize product design, and enhance customer experience.

### 1.2 Research Objectives and Significance

Against the backdrop of population aging and the development of China's pension system, this project investigates commercial pension insurance purchase behavior using machine learning techniques.

The objectives are to identify characteristics associated with insurance purchase behavior, evaluate the predictive performance of different machine learning models, and interpret the contribution of individual features to model predictions.

The findings may provide data-driven insights for customer identification, targeted marketing, and product development in the insurance industry. They may also contribute to discussions on the development of China's third-pillar pension system.

### 1.3 Research Methods

This project uses data from the Chinese General Social Survey (CGSS) and applies three ensemble learning algorithms:

- Random Forest
- XGBoost
- LightGBM

The models are evaluated and compared using classification metrics, with particular attention to recall and F1-score. SHAP analysis and partial dependence plots (PDPs) are used to interpret model predictions and explore how selected features are associated with predicted purchase probabilities.

## 2. Dataset

### 2.1 Data Source

The data come from the 2021 and 2023 waves of the Chinese General Social Survey (CGSS), a nationwide, longitudinal academic survey program that collects information on social, community, family, and individual characteristics across mainland China.

The survey covers a broad range of variables potentially associated with commercial pension insurance purchase behavior, including respondents' insurance participation, demographic characteristics, family structure, social attitudes, and living conditions.

An exploratory analysis of the 2023 survey data, which contains 11,326 observations, revealed substantial missingness in several variables required for this project. This is partly attributable to the survey's use of randomized subsampling for certain questions. For example, the numbers of missing responses for the number of sons and daughters were 4,360 and 4,359, respectively. The corresponding missing counts for new-media and traditional-media usage were 4,347 and 4,350.

Dropping all observations with missing values in these variables would substantially reduce the available sample size. Imputing the missing values could also introduce uncertainty and potentially distort relationships among variables.

To retain a larger number of observations, this project combines the 2021 and 2023 survey data. This decision reflects a trade-off between the benefits of a larger sample and potential differences between survey waves. After merging the two datasets, the combined dataset contains 19,474 observations before subsequent preprocessing.

**Data availability:** The original CGSS datasets are not included in this public repository because of data access restrictions and responsible data-sharing considerations. Please refer to the accompanying data documentation for details.

### 2.2 Variable Selection and Feature Engineering

The project uses one target variable and 22 explanatory variables.

The explanatory variables cover the following areas:

- **Demographic characteristics:** age, marital status, education level, and residential area.
- **Family characteristics:** household size, number of sons, number of daughters, and number of properties owned by the household.
- **Intergenerational relationships:** economic support, emotional support, and practical care.
- **Media use and social interaction:** traditional-media usage, new-media usage, and frequency of social interaction.
- **Economic and psychological characteristics:** annual household income, happiness, depressive feelings, self-rated health, and financial literacy.
- **Social security and attitudes:** participation in basic pension insurance, participation in basic medical insurance, and attitudes toward pension provision.

The target variable is `com_endow`, which indicates whether a respondent has purchased commercial pension insurance.

Detailed variable definitions and construction procedures are provided in `1_1_Variable_Description`. Descriptive statistics are presented in `2_x`.

## 3. Methodology

### 3.1 Handling Class Imbalance

The target variable is highly imbalanced, with substantially fewer positive observations (insurance purchasers) than negative observations (non-purchasers).

Under such conditions, machine learning models may become biased toward the majority class and fail to identify positive cases adequately.

To address this issue, the project applies the Synthetic Minority Over-sampling Technique (SMOTE) to the training set. SMOTE generates synthetic minority-class observations to increase the representation of positive cases during model training.

For model evaluation, the original test set is also downsampled by randomly removing a portion of the negative observations. This produces a test set with a more balanced class distribution, allowing the model's ability to identify positive cases to be examined more directly.

However, downsampling changes the class distribution of the test set. Consequently, evaluation metrics calculated on this modified test set may not reflect performance under the original population distribution. In particular, precision and other threshold-dependent metrics should be interpreted with this limitation in mind.

To prevent data leakage, resampling should be performed only after the train-test split, and synthetic samples should be generated exclusively from the training data.

### 3.2 Evaluation Metrics

The project evaluates binary classification performance using the confusion matrix, precision, recall, F1-score, the Receiver Operating Characteristic (ROC) curve, and the Area Under the ROC Curve (AUC).

Accuracy alone may be misleading when the target variable is highly imbalanced. Since identifying potential purchasers is a key objective in insurance marketing, particular attention is paid to recall, which measures the proportion of actual positive cases correctly identified by the model.

In this application, failing to identify a potential purchaser may represent a missed marketing opportunity, whereas contacting a person who ultimately does not purchase insurance may incur additional marketing costs. This motivates a focus on recall, while F1-score is also considered to balance precision and recall.

These trade-offs depend on actual business costs and should ultimately be validated using operational data.

## 4. Key Results

### 4.1 Model Comparison

Among the three machine learning models, XGBoost achieves the highest recall and the strongest overall performance according to the selected evaluation metrics.

XGBoost identifies approximately 72% of the actual positive cases in the evaluation sample, outperforming Random Forest and LightGBM in recall. Its precision is 0.46, indicating that 46% of the observations predicted as purchasers are actual positive cases in the evaluation sample.

Although XGBoost has lower precision than the other two models, its higher recall means that it misses fewer actual purchasers. It also achieves an F1-score of 0.56 and an AUC of 0.76. Based on these results and the project's emphasis on identifying potential purchasers, XGBoost is selected for further interpretation.

Detailed model outputs are provided in `7_0`, `7_1`, `7_2`, and `7_3`.

### 4.2 SHAP Analysis

SHAP analysis is used to examine the contribution of individual features to the model's predictions.

The ten most important features in the XGBoost model are:

1. Education level
2. Urban residence
3. Participation in basic medical insurance
4. Financial literacy
5. Self-reported depressive feelings
6. Number of properties owned by the household
7. New-media usage frequency
8. Number of daughters
9. Traditional-media usage frequency
10. Age

These results indicate that demographic, socioeconomic, health-related, and media-use characteristics contribute to the model's predictions.

SHAP importance reflects a feature's contribution to model predictions. It should not be interpreted as evidence that the feature causes insurance purchase behavior.

Detailed results are provided in `8_1_SHAP_Analysis`.

### 4.3 Partial Dependence Analysis

Partial dependence plots (PDPs) are used to explore how selected features relate to the model's average predicted probability of commercial pension insurance purchase.

The plots suggest positive associations between predicted purchase probability and certain values of education level, depressive-feeling scores, and traditional- and new-media usage. Nonlinear patterns are also observed for the number of daughters, age, and household property ownership.

These patterns describe the behavior of the fitted model rather than causal effects. Their interpretation should also account for potential correlations among explanatory variables and the limitations of partial dependence plots.

Detailed plots and interpretations are provided in `8_2_PDP_Analysis`.

## 5. Main Findings and Practical Implications

### 5.1 Model Selection and Class Imbalance

Handling class imbalance is an important part of this prediction task. In the experiments conducted for this project, applying SMOTE to the training set to achieve a 1:1 class ratio, together with randomly downsampling the test set to a positive-to-negative ratio of approximately 3:7, produced the best overall performance among the approaches tested.

Under this evaluation setup, XGBoost is the best-performing model according to the selected metrics.

These findings are specific to the current experimental design. Because test-set downsampling changes the class distribution, further evaluation on a test set that preserves the original distribution is necessary before drawing conclusions about real-world deployment performance.

### 5.2 Important Predictive Characteristics

The analysis identifies education level, residential area, basic medical insurance participation, financial literacy, household property ownership, depressive feelings, number of daughters, media usage, and age as important predictive features.

Some of these characteristics, such as age, income, property ownership, financial literacy, and residential area, are already relevant to insurance customer segmentation. Other features, including education level, self-reported depressive feelings, and media usage, may provide additional perspectives for understanding differences in predicted purchase behavior.

For insurers, these findings may inform further research into customer segmentation, communication channels, and product design. Any use of personal or sensitive characteristics in marketing should be subject to applicable privacy, data-protection, fairness, and insurance regulations.

### 5.3 Patterns in Predicted Purchase Probability

The partial dependence analysis suggests that respondents with certain characteristics—such as a university education or higher, urban residence, participation in investment activities, ownership of two or more properties, lower reported levels of negative feelings, one daughter, frequent media use, or an age between 55 and 65—may have higher predicted purchase probabilities under the fitted model.

These patterns should be interpreted as model-based associations rather than definitive rules for identifying individual customers. They do not establish that any particular characteristic causes a person to purchase commercial pension insurance.

Before applying these findings in practice, insurers would need to validate them using appropriate business data and assess their effectiveness, fairness, and regulatory compliance.

## 6. Limitations

### 6.1 Class Imbalance and Evaluation Design

The original dataset has a highly imbalanced target variable. Although SMOTE increases the representation of the minority class during training, synthetic observations may not fully capture the characteristics of real purchasers.

Furthermore, downsampling the test set changes the original class distribution and may affect the interpretation of precision, recall, and other evaluation metrics. Future work should include evaluation on a test set that preserves the original distribution and, where appropriate, use additional metrics or cost-sensitive evaluation methods.

### 6.2 Missing Variables and Data Limitations

Due to the limited coverage of available variables and substantial missingness in certain survey questions, some potentially informative features—such as intergenerational economic support, emotional support, practical care, occupation, and other household characteristics—could not be fully incorporated into the final model.

Future research could explore more comprehensive datasets, improved missing-data strategies, and alternative feature combinations to identify additional predictive patterns and improve model performance.

### 6.3 Generalizability to Real-World Insurance Operations

The features used in this project are primarily derived from an academic social survey and previous academic research. Therefore, the findings are more suitable for exploratory analysis than for direct operational deployment.

The actual predictive performance of these models in insurance marketing would need to be validated using appropriately authorized insurance-company data. Such validation would also need to account for differences between survey respondents and real-world customers, as well as changes in marketing practices and consumer behavior over time.

## 7. Reproducibility and How to Run

The repository contains the code and documentation used to demonstrate the project's analytical workflow. The original CGSS datasets are not included because of data access restrictions.

Please refer to the data documentation for information about data availability. If simulated data are provided, they are intended to demonstrate the code structure and workflow only; results obtained from simulated data should not be interpreted as empirical findings based on the original survey data.

For the complete workflow, run the relevant notebooks in the order indicated by their filenames and documentation. The notebooks contain the data-processing steps, model training procedures, evaluation outputs, visualizations, and interpretations.

Because the original data are not included, the complete empirical analysis cannot be reproduced from this public repository alone.