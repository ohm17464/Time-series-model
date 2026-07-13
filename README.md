# Time-GPT
# Time Series Forecasting Framework: Time-GPT vs. LightGBM (with TDA)

โครงงานวิจัยและพัฒนาสถาปัตยกรรมระบบพยากรณ์ข้อมูลอนุกรมเวลาความละเอียดสูง (Multi-Granular Time Series Forecasting) สำหรับข้อมูลที่มีความถี่ทุก 30 นาที โดยมุ่งเน้นการเปรียบเทียบประสิทธิภาพการทำ Day-Ahead Forecasting แบบไร้การรั่วไหลของข้อมูล (Zero Data Leakage) ระหว่างโมเดลระดับรากฐาน Time-GPT (Nixtla), LightGBM (พร้อมฟีเชอร์ TDA) และระบบสถิติอ้างอิง Powerformer Sliding Window

## คุณสมบัติเด่นของระบบ

- **Foundation Model Integration**: ประยุกต์ใช้สถาปัตยกรรมระดับรากฐานด้วย Time-GPT (ผ่าน NixtlaClient) ในการทดสอบแบบ Zero-shot Forecasting
- **Topological Data Analysis (TDA)**: สกัดคุณลักษณะพิเศษทางสถิติและคณิตศาสตร์ด้วยเทคนิค Persistence Entropy (ผ่านคลังไลบรารี giotto-tda) เพื่อเสริมมิติข้อมูลให้กับตัวแบบ LightGBM
- **Strict Data Isolation (No-Leakage)**: ออกแบบระบบพยากรณ์ล่วงหน้า 1 วัน (Day-Ahead) โดยการทำ Shift ข้อมูลย้อนหลังไปอย่างน้อย 48 steps (24 ชั่วโมง) เพื่อการันตีความโปร่งใสในการทดสอบและป้องกัน Data Leakage
- **Multi-Granular Evaluation**: ระบบประเมินผลเชิงลึกแยกตามระดับเวลา (Hourly, Daily, Weekly, Monthly) พร้อมการคำนวณแบบ RAW RMSE เพื่อป้องกันปัญหาการหักล้างของสัญญาณข้อมูล (Error Cancellation) จากการหาค่าเฉลี่ยล่วงหน้า
- **Automated Visualization**: แสดงผลลัพธ์การพยากรณ์จริงเปรียบเทียบกับตัวแบบต่างๆ พร้อมแนบตารางสรุปผลประสิทธิภาพ MAE และ RMSE ไว้ใต้กราฟโดยอัตโนมัติ เพื่อความสะดวกในการวิเคราะห์ผลงานวิจัย

## โครงสร้างตัวแบบการพยากรณ์

1. **Powerformer Sliding Window**: ตัวแบบเชิงสถิติถ่วงน้ำหนักตามระยะเวลาในอดีต (Exponential Decay) สำหรับเป็นค่ามาตรฐานอ้างอิง (Baseline)
2. **LightGBM (with TDA)**: ตัวแบบแมชชีนเลิร์นนิง Gradient Boosting ที่ประมวลผลร่วมกับข้อมูลอดีต (Lags, Rolling Mean) และคุณลักษณะทางโทโพโลยี (TDA Entropy)
3. **Time-GPT**: โมเดลโครงข่ายประสาทเทียมขนาดใหญ่ประเภท Transformer-based สำหรับข้อมูลอนุกรมเวลาโดยเฉพาะ

## การเตรียมระบบและการติดตั้ง

### 1. ไลบรารีที่จำเป็น (Dependencies)
โปรเจกต์นี้พัฒนาบนภาษา Python (แนะนำเวอร์ชัน 3.9 ขึ้นไป) โดยมีไลบรารีหลักดังนี้:

```bash
pip install numpy pandas matplotlib lightgbm nixtla python-dotenv scikit-learn
# สำหรับการใช้งานฟีเชอร์ TDA
pip install giotto-tda

NIXTLA_API_KEY=รหัส_api_key_ของคุณ

D:/Time-GPT/
├── dataset/
│   ├── KS_ready.csv
│   └── TK_ready.csv
├── project/
│   └── .env
└── research_results/     # ไดเรกทอรีสำหรับบันทึกภาพและผลการทดลอง

if __name__ == "__main__":
    # ตัวอย่างการสั่งรันระบบวิจัยเปรียบเทียบผลลัพธ์ของชุดข้อมูลเมือง KS และ TK
    run_research_v24("KS")
    run_research_v24("TK")
