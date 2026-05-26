# CLAUDE.md — สรุปโปรเจกต์เชิงเทคนิค

เอกสารนี้สรุปภาพรวมโปรเจกต์สำหรับผู้รีวิวโค้ด / HR / ผู้ที่เข้ามาอ่านครั้งแรก
(รายละเอียดวิธีใช้งานดูที่ [README.md](README.md))

---

## โปรเจกต์คืออะไร

**PlayCast** — โปรเจกต์ **ปี 2 วิชา Linear Algebra**
เป้าหมาย: นำทฤษฎี **เวกเตอร์และเมทริกซ์** มาประยุกต์ใช้จริงผ่าน Machine Learning
เพื่อ **ทำนายจำนวนผู้เป็นเจ้าของ (estimated owners) ของเกมบน Steam**

เขียนด้วย **Python** ใช้ `pandas` (จัดการข้อมูล), `scikit-learn` (โมเดล/เวกเตอร์),
`numpy` (คำนวณเชิงตัวเลข) และ `tkinter` (หน้าจอ GUI)

---

## ทำไมถึงเกี่ยวกับ Linear Algebra

โปรเจกต์นี้แสดงให้เห็นว่าแนวคิดในวิชา Linear Algebra ใช้แก้ปัญหาจริงได้อย่างไร:

1. **การสร้างเวกเตอร์จากข้อมูลข้อความ** — แต่ละเกมมีคุณสมบัติเป็นข้อความ (แท็ก, ภาษา, หมวดหมู่)
   เราแปลงเป็นเวกเตอร์ one-hot ด้วย `MultiLabelBinarizer` ทำให้เกมหนึ่งเกม = เวกเตอร์ในปริภูมิ 4,963 มิติ

2. **เมทริกซ์และการถดถอยเชิงเส้น** — เรียงเวกเตอร์ของทุกเกมเป็นเมทริกซ์ X
   แล้วใช้ Linear Regression หาเวกเตอร์สัมประสิทธิ์ β ที่ทำให้ Xβ ≈ y
   (หลักการคือ Least Squares / Normal Equation: β = (XᵀX)⁻¹Xᵀy)

3. **ความคล้ายด้วย Cosine Similarity** — วัดมุมระหว่างเวกเตอร์ฟีเจอร์ของเกมสองเกม
   เพื่อหา "เกมที่คล้ายกันที่สุด" (ใช้ dot product / norm ของเวกเตอร์)

---

## สถาปัตยกรรม / Data Pipeline

```
games.json ──▶ converted.csv ──▶ games_clean.csv ──▶ owners_model_mlr.joblib ──▶ GUI
(Kaggle)       แปลงเป็นตาราง       กรองให้เหลือเกม       โมเดลเทรนแล้ว           ทำนาย+แนะนำ
```

| ไฟล์ (`src/`) | บทบาท |
|---|---|
| `convert_json_csv.py` | อ่าน JSON จาก Kaggle, แปลง dict/list เป็น string, เซฟเป็น CSV |
| `clean_data.py` | กรองทิ้งสิ่งที่ไม่ใช่เกม (ซอฟต์แวร์/เครื่องมือ), ราคา < 1 USD, ไม่มีเวลาเล่น, ไม่มีภาษาอังกฤษ ฯลฯ |
| `playcast.py` | สร้างฟีเจอร์ (encode เวกเตอร์), เทรน Linear Regression, ฟังก์ชัน `predict_new_game` และ `find_similar_games` |
| `gui_play_cast.py` | หน้าจอ tkinter: ค้นหาเกม, เติมข้อมูลอัตโนมัติ, ปุ่มทำนาย/หาเกมคล้าย |

**โมเดลที่บันทึก** (`models/owners_model_mlr.joblib`) เก็บเป็น dict 3 ส่วน:
`model` (LinearRegression), `encoders` (MultiLabelBinarizer 4 ตัว), `feature_columns` (ลำดับคอลัมน์ 4,963 ช่อง)

---

## ฟีเจอร์ที่ใช้เทรน

- **ตัวเลข:** ราคา, ปีที่วางขาย, เวลาเล่นเฉลี่ย, log ของรีวิวบวก/ลบ, สัดส่วนรีวิวบวก
- **เวกเตอร์ one-hot:** Tags (442), Supported languages (57), Publishers (4,418), Categories (40)
- **เป้าหมาย (y):** `log1p(estimated owners)` — ทำนายแล้วแปลงกลับด้วย `expm1`

ชุดข้อมูลหลังกรอง: **~10,400 เกม**

---

## วิธีรันอย่างย่อ

```bash
pip install numpy pandas scikit-learn joblib
python src/gui_play_cast.py        # ใช้โมเดลที่มีอยู่แล้วได้เลย
```
เทรนใหม่: `convert_json_csv.py → clean_data.py → playcast.py` (ต้องมี `data/games.json` จาก Kaggle ก่อน)

---

## หมายเหตุสำหรับผู้รีวิว

- โค้ดทุกไฟล์อ้างอิง path จาก `__file__` จึงรันจาก working directory ไหนก็ได้
- ไฟล์ข้อมูลขนาดใหญ่ (`games.json` 707MB, `converted.csv` 505MB) ไม่ได้อยู่ใน git
  (โหลด/สร้างเองตาม pipeline) — ส่วน `games_clean.csv` ที่ใช้เทรนจริงเก็บไว้ใน repo
- เป็นงานเชิงการศึกษา เน้นสาธิตการประยุกต์ Linear Algebra มากกว่าความแม่นยำระดับ production
