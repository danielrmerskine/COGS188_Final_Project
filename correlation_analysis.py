import pandas as pd

file = "prostate_cancer_prediction.csv"
df = pd.read_csv(file)

drop = ["Patient_ID", "Follow_Up_Required", "Screening_Age", "Previous_Cancer_History", "Early_Detection"]
cleanDf = df.drop(columns=drop, errors="ignore")

encodedDf = cleanDf.copy()

binaryCols = [
    "Family_History", "Race_African_Ancestry", "Difficulty_Urinating",
    "Weak_Urine_Flow", "Blood_in_Urine", "Pelvic_Pain", "Exercise_Regularly", 
    "Healthy_Diet", "Smoking_History", "Hypertension", 
    "Diabetes", "Genetic_Risk_Factors"
]

for col in binaryCols:
    if col in encodedDf.columns:
        encodedDf[col] = encodedDf[col].map({"Yes": 1, "No": 0})

encodedDf["Biopsy_Result"] = encodedDf["Biopsy_Result"].map({"Malignant": 1, "Benign": 0})

if "DRE_Result" in encodedDf.columns:
    encodedDf["DRE_Result"] = encodedDf["DRE_Result"].map({"Normal": 0, "Abnormal": 1})

if "Alcohol_Consumption" in encodedDf.columns:
    encodedDf["Alcohol_Consumption"] = encodedDf["Alcohol_Consumption"].map({"Low": 1, "Moderate": 2, "High": 3})

if "Cholesterol_Level" in encodedDf.columns:
    encodedDf["Cholesterol_Level"] = encodedDf["Cholesterol_Level"].map({"Normal": 0, "High": 1})

biopsyCorrelations = encodedDf.corr(numeric_only=True)["Biopsy_Result"].sort_values(ascending=False)

print("Ranked Feature Correlations with Biopsy Result:")
print(biopsyCorrelations)
