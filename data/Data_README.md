# 数据来源

本项目使用中国综合社会调查（Chinese General Social Survey，CGSS）2021 年和 2023 年两期调查数据。CGSS 是一项全国性社会调查，涵盖人口统计特征、家庭结构、社会态度和生活状况等多个方面。

本项目旨在探索与商业养老保险购买行为相关的特征，并利用机器学习方法构建预测模型。

# 数据可用性

出于数据使用权限及负责任的数据共享考虑，本公开仓库不包含原始 CGSS 调查数据。

有意使用原始数据的研究者，请通过 CGSS 官方网站及相关数据访问平台了解数据获取方式，并遵守适用的数据使用条款。

# 数据处理与特征工程

本项目对调查数据进行了数据清洗、缺失值处理和特征构造，以便开展机器学习分析。构建的特征主要包括以下类别：

人口统计特征

家庭结构特征

媒体使用与社会互动特征

健康相关特征

社会态度及其他潜在相关因素

目标变量 com_endow 用于表示受访者是否购买商业养老保险，具体定义与编码方式以本项目的数据处理规则为准。

# 可复现性说明

由于本仓库不包含原始调查数据，用户无法仅凭仓库中的文件完整复现基于原始 CGSS 数据开展的全部分析。

项目可能提供模拟数据，用于演示数据处理流程、特征工程、模型训练及评估方法。基于模拟数据得到的结果仅用于展示代码实现流程，不应被视为基于原始 CGSS 调查数据得到的实证结论。

# 数据使用与引用

使用原始 CGSS 数据前，请查阅官方数据说明，了解数据获取条件、允许用途及引用要求。所有数据使用行为均应遵守适用的数据使用条款。

有关项目的研究方法、代码实现和模型评估，请参阅项目根目录下的 README.md 及相关 Jupyter Notebook。

# Data Source

This project uses data from the Chinese General Social Survey (CGSS), specifically the 2021 and 2023 survey waves. CGSS is a nationwide social survey covering a broad range of topics, including demographic characteristics, family structure, social attitudes, and living conditions.

The project aims to explore factors associated with commercial pension insurance purchase behavior and develop machine learning models for predictive analysis.

# Data Availability

The original CGSS datasets are not included in this public repository due to data access restrictions and responsible data-sharing considerations.

Users interested in the original datasets should consult the official CGSS website and the relevant data access platform, and comply with all applicable terms of use.

# Data Processing and Feature Engineering

The survey data were processed to construct model features, handle missing values, and prepare the data for machine learning. The features cover several categories, including:

Demographic characteristics

Family structure

Media usage and social interaction

Health-related characteristics

Attitudes and other potentially relevant factors

The target variable, com_endow, indicates whether a respondent has purchased commercial pension insurance, according to the project's data processing and coding rules.

# Reproducibility

Because the original survey data are not included in this repository, the complete analysis cannot be reproduced using the repository alone.

Simulated data may be provided to demonstrate the data processing pipeline, feature engineering, model training, and evaluation procedures. Results generated from simulated data are for demonstration purposes only and should not be interpreted as empirical findings based on the original CGSS data.

# Data Usage and Citation

Users should consult the official CGSS documentation for data access conditions, permitted uses, and citation requirements. Any use of the original survey data must comply with the applicable terms and conditions.

For details on the project methodology and implementation, please refer to the main README.md and the Jupyter notebooks.