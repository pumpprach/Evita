import json
from huggingface_hub import InferenceClient
from openai import OpenAI

class QwenExtractor:
    def __init__(self, token):
        # self.client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )

    def __call__(self, note):
        target_features = [
            "Family_Diabetes", "Smoker", "Alcohol", "Hypertension", 
            "Prediabetes", "Obesity", "High_Cholesterol", "Fatty_Liver", 
            "Sedentary_Lifestyle", "Poor_Diet"
        ]
        
        system_prompt = (
            f"You are a clinical data extractor. Read the clinical note and return a JSON object. "
            f"Extract exactly these keys: {', '.join(target_features)}. "
            "Definitions: Prediabetes (high sugar/HbA1c), Obesity (high BMI), Sedentary (lack of exercise), Poor_Diet (junk food/sweets). "
            "Use 1 for Yes, 0 for No. If NOT mentioned, leave out of the JSON."
        )

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Clinical Note: {note}"}
            ],
            temperature=0.1, # ปรับให้ต่ำเพื่อเน้นความแม่นยำของ JSON
            max_tokens=500
        )
        
        # ดึงข้อความออกมา
        content = response.choices[0].message.content
        
        # ทำความสะอาดข้อมูล (เผื่อ LLM แถม ```json มา)
        clean_response = content.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            print("LLM Error: Could not parse JSON")
            return {}