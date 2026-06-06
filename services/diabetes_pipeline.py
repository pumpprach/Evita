import pandas as pd
import numpy as np
import xgboost as xgb
import shap

FEATURE_NAMES = ["Resting_Heart_Rate", "Sleep_HRV_Avg", "Daily_Spot_SpO2", "Sleep_Duration", "Sleep_Efficiency", "Sleep_Regularity_Index", 
       "Total_Daily_Steps", "Sedentary_Hours", "Age", "Gender", "Family_Diabetes", "Smoker", "Alcohol", "Hypertension", "Prediabetes", 
       "Obesity", "High_Cholesterol", "Fatty_Liver", "Sedentary", "Poor_Diet", "Resting_Heart_Rate_patient_scaled", "Sleep_HRV_Avg_patient_scaled", 
       "Daily_Spot_SpO2_patient_scaled", "Steps_RollMean_3", "HRV_RollMean_3", "SpO2_RollMean_3"]

class DiabetesRiskPipeline:
    def __init__(self, model_path, llm_extractor_func):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.llm_extractor = llm_extractor_func
        self.feature_names = FEATURE_NAMES

    def _extract_notes(self, df):
        """สกัดข้อมูลจาก Clinical_Notes โดยใช้ LLM"""
        # เอาเฉพาะค่าที่ไม่ซ้ำเพื่อลดการเรียก LLM ซ้ำซ้อน (ถ้ามี)
        extracted_data = {}
        unique_notes = df['Clinical_Notes'].unique()
        
        for note in unique_notes:
            if pd.isna(note) or note in ["", "ไม่มีข้อมูลเพิ่มเติม", "ไม่ระบุ"]:
                extracted_data[note] = {feat: 0 for feat in ["Family_Diabetes", "Smoker", "Alcohol", "Hypertension", "Prediabetes", "Obesity", "High_Cholesterol", "Fatty_Liver", "Sedentary", "Poor_Diet"]}
            else:
                extracted_data[note] = self.llm_extractor(note)
        
        # Map กลับเข้า DataFrame
        notes_df = pd.DataFrame(extracted_data).T
        return df.join(notes_df, on='Clinical_Notes')

    def _engineer_features(self, df):
        """จัดการ Rolling Mean, Scaling และเติมคอลัมน์ให้ครบ"""
        df = df.copy()
        
        # 1. Engineering ข้อมูลดิบ
        df['Total_Active_Minutes'] = df.get('Moderate_Activity_Minutes', 0) + df.get('Vigorous_Activity_Minutes', 0)
        df['Sedentary_Hours'] = df.get('Sedentary_Minutes', 0) / 60.0
        
        # 2. Rolling Mean (ใช้ .groupby ตาม Patient_ID)
        rolling_cols = {'Resting_HR': 'Resting_Heart_Rate', 'Systolic_BP': 'Systolic_BP', 
                        'Sleep_Duration': 'Sleep_Duration', 'Total_Active_Minutes': 'Total_Active_Minutes'}
        
        for col, target_name in rolling_cols.items():
            if col in df.columns:
                df[f'{target_name}_RollMean_3'] = df.groupby('Patient_ID')[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # 3. Patient Scaling (เปรียบเทียบกับค่าเฉลี่ยตัวเอง)
        for col in ['Resting_HR', 'Sleep_Duration']:
            if col in df.columns:
                mean = df.groupby('Patient_ID')[col].transform('mean')
                std = df.groupby('Patient_ID')[col].transform('std')
                df[f'{col}_patient_scaled'] = (df[col] - mean) / (std + 1e-5)
        
        # 4. บังคับ Reindex ให้คอลัมน์เรียงตาม model requirement
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = np.nan
                
        return df[self.feature_names]

    def process_and_predict(self, user_data_json):
        # 1. แปลงเป็น DataFrame
        df = pd.DataFrame([user_data_json])
        
        # 2. Extract ฟีเจอร์จาก Notes
        df = self._extract_notes(df)
        
        # 3. จัดการ Engineering (Rolling/Scaling)
        df_processed = self._engineer_features(df)
        
        # 4. ดึงแถวล่าสุดเพื่อทำนาย
        X_pred = df_processed.iloc[[-1]]
        
        # 5. ทำนาย
        prob = self.model.predict_proba(X_pred)[0]
        print(prob)
        risk_score = float((prob[1] * 0.5 + prob[2] * 1.0) * 100)
        
        # shap_values = self.explainer.shap_values(X_pred)
        
        # # กรณี SHAP return เป็น list (XGBoost Classifier มักได้เป็น list ของ class)
        # # เราต้องดึง array ของ class ที่เราสนใจออกมา แล้วดึง index [0] ออกมาอีกที
        # if isinstance(shap_values, list):
        #     target_shap = shap_values[2][0] # นี่คือ array ของค่า impact
        # else:
        #     target_shap = shap_values[0]

        # feature_impacts = []
        # for feat_name, shap_val in zip(self.feature_names, target_shap):
        #     # ใช้ .item() เพื่อแปลง numpy scalar เป็น float ของ Python
        #     impact_val = float(shap_val) if not isinstance(shap_val, np.ndarray) else float(shap_val.item())
        #     feature_impacts.append({
        #         "feature": feat_name,
        #         "impact": impact_val
        #     })
            
        # feature_impacts = sorted(feature_impacts, key=lambda x: x["impact"], reverse=True)

        return {
            "risk_score": round(risk_score, 2),
            # "top_factors": feature_impacts[:3],
            # "recommendation": "ควรเพิ่มการออกกำลังกาย" if risk_score > 50 else "สุขภาพดีเยี่ยม"
        }