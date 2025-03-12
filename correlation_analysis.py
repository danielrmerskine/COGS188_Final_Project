import pandas as pd

file = "prostate_cancer_prediction.csv"
df = pd.read_csv(file)

drop = ["Patient_ID", "Screening_Age"]
cleanDf = df.drop(columns=drop, errors="ignore")

encodedDf = cleanDf.copy()

binaryCols = [
    "Family_History", "Race_African_Ancestry", "Difficulty_Urinating", "Weak_Urine_Flow", 
    "Blood_in_Urine", "Pelvic_Pain", "Exercise_Regularly", "Healthy_Diet", "Smoking_History", 
    "Hypertension", "Diabetes", "Genetic_Risk_Factors", "Previous_Cancer_History", "Early_Detection", 
    "Survival_5_Years", "Back_Pain", "Erectile_Dysfunction", "Follow_Up_Required"
]

for col in binaryCols:
    if col in encodedDf.columns:
        encodedDf[col] = encodedDf[col].map({"Yes": 1, "No": 0})

encodedDf["Biopsy_Result"] = encodedDf["Biopsy_Result"].map({"Malignant": 1, "Benign": 0})

encodedDf["DRE_Result"] = encodedDf["DRE_Result"].map({"Normal": 0, "Abnormal": 1})

encodedDf["Alcohol_Consumption"] = encodedDf["Alcohol_Consumption"].map({"Low": 1, "Moderate": 2, "High": 3})

encodedDf["Cholesterol_Level"] = encodedDf["Cholesterol_Level"].map({"Normal": 0, "High": 1})

encodedDf["Cancer_Stage"] = encodedDf["Cancer_Stage"].map({"Localized": 1, "Advanced": 2, "Metastatic": 3})

encodedDf["Treatment_Recommended"] = encodedDf["Treatment_Recommended"].map({"Active Surveillance": 1, "Hormone Therapy": 2, "Immunotherapy": 3, "Radiation": 4, "Surgery": 5})

biopsyCorrelations = encodedDf.corr(numeric_only=True)["Biopsy_Result"].sort_values(ascending=False)

print("Ranked Feature Correlations with Biopsy Result:")
print(biopsyCorrelations)
