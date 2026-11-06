Dưới đây là bộ tài liệu được thiết kế theo hướng **đủ chuyên nghiệp để làm ĐACN2**, nhưng vẫn giữ phạm vi phù hợp với **nhóm 2–3 sinh viên trong 10–12 tuần**. Tôi cũng tách rõ phần **MVP bắt buộc** và các tính năng mở rộng để tránh dự án bị quá tải.

# TÀI LIỆU 1 — PRODUCT REQUIREMENTS DOCUMENT (PRD)

# FallGuard AI

## Hệ thống phát hiện té ngã cho người cao tuổi

**Phiên bản:** 1.0
**Thời gian dự kiến:** 10/2026 – 12/2026
**Loại dự án:** Đồ án chuyên ngành 2 + Sản phẩm dự thi AI4Life 2026
**Quy mô nhóm:** 2–3 sinh viên
**Trạng thái:** Draft

---

## 1. Tổng quan

### 1.1. Giới thiệu

**FallGuard AI** là hệ thống sử dụng Computer Vision và Deep Learning để tự động phát hiện hành vi té ngã của người cao tuổi thông qua camera.

Thay vì yêu cầu người cao tuổi phải đeo smartwatch, vòng tay hoặc thiết bị cảm biến, hệ thống sử dụng camera có sẵn trong môi trường để quan sát chuyển động. Dữ liệu hình ảnh được xử lý bằng **MediaPipe Pose** nhằm trích xuất 33 điểm mốc cơ thể. Các điểm mốc này được biểu diễn thành chuỗi dữ liệu theo thời gian và đưa vào mô hình **LSTM/BiLSTM** để phân biệt giữa hoạt động bình thường và hành vi té ngã.

Khi phát hiện một sự kiện có khả năng là té ngã, FallGuard AI tạo cảnh báo và gửi đến người chăm sóc thông qua Telegram hoặc các kênh thông báo được tích hợp.

Một nguyên tắc quan trọng của hệ thống là **Privacy by Design**: hệ thống ưu tiên xử lý dữ liệu pose/skeleton thay vì lưu trữ video gốc.

### 1.2. Vấn đề cần giải quyết

Người cao tuổi, đặc biệt là những người sống một mình hoặc cần được chăm sóc thường xuyên, có nguy cơ gặp sự cố té ngã mà không được phát hiện kịp thời.

Các phương pháp giám sát truyền thống có một số hạn chế:

* Cần người trực tiếp theo dõi.
* Camera thông thường tạo ra lượng lớn dữ liệu video.
* Thiết bị đeo có thể gây bất tiện.
* Người cao tuổi có thể quên đeo hoặc sạc thiết bị.
* Khó giám sát liên tục trong thời gian dài.

FallGuard AI hướng tới việc giải quyết vấn đề bằng một hệ thống giám sát tự động, không xâm lấn và có khả năng cảnh báo theo thời gian gần thực.

---

## 2. Mục tiêu sản phẩm

### 2.1. Mục tiêu chính

Xây dựng một prototype có khả năng:

1. Nhận dữ liệu video từ camera.
2. Phát hiện và theo dõi tư thế người.
3. Trích xuất 33 keypoint bằng MediaPipe Pose.
4. Xây dựng chuỗi dữ liệu chuyển động theo thời gian.
5. Phân loại hành động bằng LSTM/BiLSTM.
6. Phát hiện sự kiện có khả năng là té ngã.
7. Gửi cảnh báo đến người chăm sóc.
8. Hiển thị trạng thái hệ thống trên giao diện.
9. Hạn chế lưu trữ video gốc nhằm tăng tính riêng tư.
10. Cung cấp dữ liệu và kết quả đủ rõ ràng để phục vụ báo cáo và demo cuộc thi.

### 2.2. Mục tiêu học thuật

Dự án giúp nhóm áp dụng kiến thức về:

* Computer Vision.
* Deep Learning.
* Machine Learning.
* Xử lý dữ liệu chuỗi thời gian.
* Backend API.
* Frontend.
* Software Engineering.
* Testing.
* Git/GitHub.
* Triển khai một hệ thống AI hoàn chỉnh.

### 2.3. Mục tiêu sản phẩm

Sản phẩm cuối cùng cần thể hiện được chuỗi:

**Camera → Pose Estimation → Temporal Modeling → Fall Detection → Alert → Dashboard**

---

## 3. User Personas

| Persona                   | Đặc điểm                                | Nhu cầu                                 |
| ------------------------- | --------------------------------------- | --------------------------------------- |
| Người cao tuổi            | Có nguy cơ té ngã, có thể sống một mình | Được giám sát mà không cần đeo thiết bị |
| Người thân                | Không thể theo dõi liên tục             | Nhận cảnh báo nhanh khi có sự cố        |
| Điều dưỡng                | Theo dõi nhiều người                    | Quản lý nhiều khu vực/người             |
| Quản trị viên             | Quản lý hệ thống                        | Theo dõi trạng thái camera và cảnh báo  |
| Sinh viên/nhóm phát triển | Xây dựng prototype                      | Dễ phát triển, kiểm thử và trình diễn   |

### 3.1. Primary User

**Người chăm sóc/người thân** là đối tượng sử dụng chính của hệ thống cảnh báo.

Người cao tuổi là **đối tượng được bảo vệ**, không nhất thiết trực tiếp tương tác với phần mềm.

---

# 4. Phạm vi sản phẩm

## 4.1. In Scope

Phiên bản MVP bao gồm:

* Nhận video từ webcam/camera.
* Phát hiện người.
* Pose estimation bằng MediaPipe.
* Chuẩn hóa keypoint.
* Xây dựng sequence.
* Phân loại Fall / Normal.
* Hiển thị trạng thái.
* Cảnh báo khi phát hiện té ngã.
* Telegram Bot.
* Lưu lịch sử sự kiện.
* Dashboard cơ bản.
* Đánh giá Accuracy, Precision, Recall, F1-score.
* Kiểm thử trên video mẫu.

## 4.2. Out of Scope

Để đảm bảo tính khả thi, MVP **không bao gồm**:

* Chẩn đoán y khoa.
* Thay thế bác sĩ/điều dưỡng.
* Nhận diện bệnh lý.
* Nhận diện danh tính khuôn mặt.
* Theo dõi nhiều camera phức tạp.
* Hệ thống cấp cứu chuyên nghiệp.
* Triển khai production-scale tại bệnh viện.
* Lưu trữ video dài hạn.
* Phân tích sức khỏe toàn diện.

---

# 5. Yêu cầu chức năng

### Quy ước ưu tiên

* **Must-have:** Bắt buộc cho MVP.
* **Should-have:** Nên có nếu đủ thời gian.
* **Could-have:** Tính năng mở rộng.

| ID    | Chức năng            | Mô tả                               | Priority    |
| ----- | -------------------- | ----------------------------------- | ----------- |
| FR-01 | Camera Input         | Nhận video từ webcam/video file     | Must-have   |
| FR-02 | Person Detection     | Xác định người trong frame          | Must-have   |
| FR-03 | Pose Estimation      | Trích xuất 33 keypoint              | Must-have   |
| FR-04 | Data Preprocessing   | Chuẩn hóa dữ liệu pose              | Must-have   |
| FR-05 | Sequence Generation  | Tạo chuỗi keypoint theo thời gian   | Must-have   |
| FR-06 | Fall Classification  | Phân loại Fall/Normal               | Must-have   |
| FR-07 | Confidence Score     | Hiển thị xác suất dự đoán           | Must-have   |
| FR-08 | Fall Event Detection | Xác định một sự kiện té ngã         | Must-have   |
| FR-09 | Alert                | Gửi cảnh báo                        | Must-have   |
| FR-10 | Alert History        | Lưu lịch sử cảnh báo                | Must-have   |
| FR-11 | Dashboard            | Hiển thị trạng thái camera/hệ thống | Must-have   |
| FR-12 | Event Detail         | Hiển thị thời gian và confidence    | Should-have |
| FR-13 | Multiple Camera      | Hỗ trợ nhiều camera                 | Could-have  |
| FR-14 | SMS Alert            | Gửi SMS                             | Could-have  |
| FR-15 | Mobile App           | Ứng dụng mobile                     | Could-have  |
| FR-16 | Analytics            | Thống kê số sự kiện                 | Should-have |
| FR-17 | Privacy Mode         | Chỉ hiển thị skeleton               | Must-have   |
| FR-18 | System Health        | Kiểm tra trạng thái hệ thống        | Should-have |

---

## 5.1. FR-01 — Camera Input

Hệ thống phải hỗ trợ:

* Webcam.
* Video file dùng cho kiểm thử.
* Camera IP nếu điều kiện cho phép.

MVP ưu tiên webcam hoặc video file nhằm giảm độ phức tạp.

---

## 5.2. FR-03 — Pose Estimation

MediaPipe Pose được sử dụng để trích xuất:

* 33 landmark.
* Tọa độ x.
* Tọa độ y.
* Tọa độ z.
* Visibility nếu có.

Dữ liệu được chuyển thành vector đặc trưng phục vụ mô hình Deep Learning.

---

## 5.3. FR-04 — Preprocessing

Dữ liệu pose cần được chuẩn hóa để giảm ảnh hưởng của:

* Khoảng cách tới camera.
* Vị trí người trong frame.
* Chiều cao khác nhau.
* Scale cơ thể.

Có thể sử dụng:

* Center normalization.
* Scale normalization.
* Missing-keypoint handling.
* Smoothing.

---

## 5.4. FR-05 — Sequence Generation

Thay vì phân tích một frame riêng lẻ, hệ thống tạo sequence:

```text
Frame 1
   ↓
Frame 2
   ↓
...
Frame N
   ↓
Pose Sequence
   ↓
LSTM/BiLSTM
```

Điều này cho phép mô hình học **chuyển động theo thời gian**, thay vì chỉ nhìn tư thế tại một thời điểm.

---

## 5.5. FR-06 — Fall Classification

Mô hình nhận sequence pose và trả về:

```text
Normal
Fall
```

Kèm confidence score.

Ví dụ:

```text
Prediction: FALL
Confidence: 94.7%
```

---

## 5.6. FR-08 — Fall Event Detection

Không gửi cảnh báo chỉ dựa trên một frame duy nhất.

Hệ thống nên sử dụng cơ chế:

```text
Prediction
     ↓
Confidence threshold
     ↓
Temporal confirmation
     ↓
Fall Event
     ↓
Alert
```

Mục tiêu là giảm false positive.

---

## 5.7. FR-09 — Alert

MVP sử dụng Telegram Bot.

Thông báo tối thiểu:

```text
⚠️ FALL DETECTED

Time: 14:32:18
Confidence: 94.7%
Camera: Room 01
Status: Requires attention
```

---

## 5.8. FR-17 — Privacy Mode

Hệ thống ưu tiên:

* Không lưu video gốc.
* Không nhận diện khuôn mặt.
* Không lưu danh tính người.
* Chỉ lưu pose hoặc metadata sự kiện.

Ví dụ dữ liệu lưu:

```text
timestamp
camera_id
prediction
confidence
event_id
```

---

# 6. Yêu cầu phi chức năng

| ID     | Nhóm            | Yêu cầu                                          | Mức    |
| ------ | --------------- | ------------------------------------------------ | ------ |
| NFR-01 | Performance     | Xử lý gần real-time trên máy demo                | Must   |
| NFR-02 | Latency         | Cảnh báo trong vài giây sau khi xác nhận sự kiện | Must   |
| NFR-03 | Accuracy        | F1-score mục tiêu ≥ 90% trên test set            | Should |
| NFR-04 | Privacy         | Không lưu video gốc mặc định                     | Must   |
| NFR-05 | Security        | API có authentication cơ bản                     | Should |
| NFR-06 | Availability    | Hệ thống chạy ổn định trong phiên demo           | Must   |
| NFR-07 | Usability       | Dashboard dễ sử dụng                             | Must   |
| NFR-08 | Scalability     | Có khả năng mở rộng nhiều camera                 | Could  |
| NFR-09 | Maintainability | Code được chia module                            | Must   |
| NFR-10 | Reproducibility | Có requirements và hướng dẫn setup               | Must   |

---

# 7. Kiến trúc hệ thống

## 7.1. Kiến trúc tổng thể

```text
             CAMERA
                │
                ▼
        ┌─────────────────┐
        │   OpenCV        │
        │ Video Capture   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ MediaPipe Pose  │
        │ 33 Landmarks    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Preprocessing   │
        │ Normalization   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Sequence Buffer │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ LSTM / BiLSTM   │
        └────────┬────────┘
                 │
          ┌──────┴───────┐
          ▼              ▼
       NORMAL           FALL
                           │
                           ▼
                  ┌────────────────┐
                  │ Event Manager  │
                  └───────┬────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        FastAPI Backend           Telegram Bot
              │
              ▼
        Dashboard/UI
```

---

## 7.2. Data Flow

### Bước 1 — Capture

Camera cung cấp frame.

### Bước 2 — Pose Estimation

MediaPipe phát hiện 33 landmark.

### Bước 3 — Feature Extraction

Các landmark được chuyển thành vector số.

### Bước 4 — Temporal Sequence

Các frame liên tiếp được ghép thành sequence.

### Bước 5 — Classification

LSTM/BiLSTM phân loại sequence.

### Bước 6 — Event Confirmation

Hệ thống kiểm tra threshold và temporal consistency.

### Bước 7 — Alert

Nếu xác nhận Fall:

```text
Fall Event
    ↓
FastAPI
    ↓
Telegram Bot
    ↓
Caregiver
```

---

# 8. Technology Stack

| Thành phần      | Công nghệ                 |
| --------------- | ------------------------- |
| Language        | Python                    |
| Computer Vision | OpenCV                    |
| Pose Estimation | MediaPipe Pose            |
| Deep Learning   | TensorFlow/Keras          |
| Model           | LSTM/BiLSTM               |
| Backend         | FastAPI                   |
| Frontend MVP    | Streamlit                 |
| Notification    | Telegram Bot API          |
| Experiment      | Google Colab              |
| Version Control | Git/GitHub                |
| Data            | Pose sequence             |
| Deployment      | Local/Cloud tùy điều kiện |

### Lựa chọn Streamlit cho MVP

Đối với nhóm sinh viên 2–3 người, **Streamlit được ưu tiên hơn React ở phiên bản đầu** vì:

* Phát triển nhanh.
* Không cần xây dựng frontend/backend phức tạp ngay từ đầu.
* Phù hợp demo AI.
* Dễ tích hợp Python model.

React có thể được xem là hướng mở rộng nếu nhóm có đủ thời gian.

---

# 9. Dataset và dữ liệu

## 9.1. Nguồn dữ liệu

Có thể kết hợp:

* Dataset public về fall detection.
* Dataset hành động thông thường.
* Video tự thu thập trong điều kiện kiểm soát.

Dữ liệu cần được chuyển thành pose sequence trước khi training.

## 9.2. Hai nhóm chính

```text
NORMAL
├── Walking
├── Sitting
├── Standing
├── Lying
├── Bending
└── Other daily activities

FALL
├── Forward fall
├── Backward fall
├── Side fall
└── Sit/fall transition
```

Không nên chỉ train với Fall và Non-Fall quá đơn giản nếu muốn giảm false positive trong demo.

---

# 10. Success Metrics

## 10.1. Classification Metrics

Các chỉ số chính:

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-score

```text
F1 = 2 × Precision × Recall
             ─────────────────
             Precision + Recall
```

Trong bài toán cảnh báo té ngã, **Recall và F1-score** cần được chú ý đặc biệt vì bỏ sót một sự kiện té ngã có thể nghiêm trọng hơn việc tạo một cảnh báo nhầm.

Tuy nhiên, hệ thống cũng phải kiểm soát False Positive để tránh tạo quá nhiều cảnh báo không cần thiết.

---

## 10.2. Mục tiêu định lượng

| Metric                         |     Target MVP |
| ------------------------------ | -------------: |
| Accuracy                       |          ≥ 90% |
| Precision                      |          ≥ 85% |
| Recall                         |          ≥ 90% |
| F1-score                       |          ≥ 87% |
| Alert latency                  |       ≤ 5 giây |
| System availability trong demo |          ≥ 95% |
| Video processing               | Near real-time |

Các con số trên là **mục tiêu dự án**, không phải kết quả được đảm bảo trước khi thực nghiệm.

---

# 11. Rủi ro và giả định

| Risk                      | Probability | Impact    | Mitigation                              |
| ------------------------- | ----------- | --------- | --------------------------------------- |
| Dataset không đủ          | Medium      | High      | Kết hợp public dataset + dữ liệu tự thu |
| Model overfitting         | Medium      | High      | Data augmentation + validation          |
| False positive cao        | High        | High      | Temporal confirmation                   |
| False negative            | Medium      | Very High | Tối ưu Recall                           |
| MediaPipe không nhận diện | Medium      | Medium    | Xử lý missing keypoint                  |
| Máy yếu                   | Medium      | Medium    | Giảm resolution/FPS                     |
| Telegram lỗi              | Low         | Medium    | Logging + retry                         |
| Internet mất              | Medium      | Medium    | Local detection + queue                 |
| Frontend chậm             | Low         | Medium    | Streamlit MVP                           |
| Phạm vi quá lớn           | High        | High      | Ưu tiên MVP                             |
| Không đủ thời gian        | High        | High      | Freeze feature sau tuần 7               |

---

# 12. Assumptions

Dự án giả định:

1. Camera có góc nhìn đủ để quan sát toàn bộ hoặc phần lớn cơ thể.
2. Người cần theo dõi nằm trong vùng quan sát.
3. Điều kiện ánh sáng tương đối ổn định.
4. Có máy tính đủ khả năng chạy inference.
5. Telegram API có thể được sử dụng trong môi trường demo.
6. Dataset đủ đại diện cho các tình huống thử nghiệm.
7. Đây là **prototype nghiên cứu**, không phải thiết bị y tế được chứng nhận.

---

# 13. Release Plan

## Version 0.1 — Research Prototype

**Thời gian:** Tuần 1–3

Bao gồm:

* Dataset.
* MediaPipe Pose.
* Pose extraction.
* Baseline model.

---

## Version 0.5 — AI MVP

**Thời gian:** Tuần 4–6

Bao gồm:

* LSTM/BiLSTM.
* Fall classification.
* Evaluation.
* Model optimization.

---

## Version 0.8 — Integrated MVP

**Thời gian:** Tuần 7–9

Bao gồm:

* FastAPI.
* Dashboard.
* Alert.
* Event logging.
* Privacy mode.

---

## Version 1.0 — Final Demo

**Thời gian:** Tuần 10–12

Bao gồm:

* Hoàn thiện UI.
* Testing.
* Documentation.
* Demo video.
* Slide tiếng Anh.
* Báo cáo.
* Chuẩn bị AI4Life.

---

# 14. Ghi chú

Các nội dung cần nhóm bổ sung:

* [ ] Tên thành viên.
* [ ] GVHD.
* [ ] Dataset chính thức.
* [ ] Camera sử dụng.
* [ ] Ngưỡng confidence.
* [ ] Sequence length.
* [ ] FPS.
* [ ] Model architecture cuối cùng.
* [ ] Kết quả thực nghiệm.
* [ ] Kênh cảnh báo cuối cùng.
* [ ] Tiêu chí cụ thể của AI4Life 2026.

---

# PROJECT PLAN — FALLGUARD AI

## Kế hoạch triển khai dự án

**Thời gian:** 10/2026 – 12/2026
**Thời lượng:** 12 tuần
**Nhân sự:** 3 người
**Phương pháp:** Agile/Scrum rút gọn

---

# 1. Mục tiêu dự án

Mục tiêu của dự án là hoàn thành một prototype FallGuard AI có khả năng:

```text
Camera
   ↓
Pose Estimation
   ↓
Pose Sequence
   ↓
LSTM/BiLSTM
   ↓
Fall Detection
   ↓
Alert
   ↓
Dashboard
```

Sản phẩm phải đáp ứng đồng thời:

* Yêu cầu của ĐACN2.
* Yêu cầu demo sản phẩm AI.
* Có kết quả đánh giá định lượng.
* Có mã nguồn.
* Có báo cáo.
* Có slide tiếng Anh.
* Có video demo.

---

# 2. Scope of Work

## 2.1. AI/ML

* Khảo sát bài toán.
* Chuẩn bị dataset.
* Pose extraction.
* Data preprocessing.
* Xây dựng baseline.
* LSTM/BiLSTM.
* Training.
* Evaluation.
* Model optimization.
* Export model.

## 2.2. Backend

* API.
* Model inference service.
* Event manager.
* Alert service.
* Logging.
* Database/local storage.
* Telegram integration.

## 2.3. Frontend/QA/Documentation

* Dashboard.
* Camera view.
* Prediction display.
* Event history.
* System status.
* UI testing.
* Integration testing.
* Báo cáo.
* Slide.
* Demo video.

---

# 3. WBS — Work Breakdown Structure

```text
1. Project Management
│
├── 1.1 Requirement
├── 1.2 Planning
├── 1.3 Git/GitHub
└── 1.4 Progress Tracking
│
2. AI/ML
│
├── 2.1 Dataset
├── 2.2 Pose Extraction
├── 2.3 Preprocessing
├── 2.4 Baseline Model
├── 2.5 LSTM/BiLSTM
├── 2.6 Training
├── 2.7 Evaluation
└── 2.8 Optimization
│
3. Backend
│
├── 3.1 API Design
├── 3.2 Inference Service
├── 3.3 Event Manager
├── 3.4 Alert Service
├── 3.5 Logging
└── 3.6 Integration
│
4. Frontend
│
├── 4.1 Dashboard
├── 4.2 Camera View
├── 4.3 Prediction UI
├── 4.4 Alert History
└── 4.5 System Status
│
5. Testing
│
├── 5.1 Unit Test
├── 5.2 Model Test
├── 5.3 Integration Test
├── 5.4 Performance Test
└── 5.5 User Scenario Test
│
6. Documentation
│
├── 6.1 Report
├── 6.2 Slides
├── 6.3 Demo Video
└── 6.4 Final Presentation
```

---

# 4. Timeline

## Tổng quan 12 tuần

| Tuần | AI/ML            | Backend           | Frontend/QA/Docs    | Milestone |
| ---- | ---------------- | ----------------- | ------------------- | --------- |
| W1   | Research         | Architecture      | Requirement         | M1        |
| W2   | Dataset          | API design        | UI design           |           |
| W3   | Pose extraction  | Backend skeleton  | Dashboard prototype | M2        |
| W4   | Baseline         | Inference API     | Camera UI           |           |
| W5   | LSTM             | Model integration | Prediction UI       | M3        |
| W6   | BiLSTM           | Event manager     | Alert UI            |           |
| W7   | Evaluation       | Telegram          | Event history       | M4        |
| W8   | Optimization     | Integration       | QA                  |           |
| W9   | Final model      | Full integration  | Full dashboard      | M5        |
| W10  | Testing          | Performance       | Documentation       |           |
| W11  | Bug fixing       | Deployment        | Report + slides     | M6        |
| W12  | Final evaluation | Demo              | Final video         | M7        |

---

# 5. Milestones

## M1 — Project Definition

**Cuối W1**

Deliverables:

* PRD.
* Project Plan.
* Architecture draft.
* Git repository.

---

## M2 — Pose Pipeline

**Cuối W3**

Deliverables:

* Video input.
* MediaPipe Pose.
* Pose dataset.
* Visualization.

---

## M3 — AI Prototype

**Cuối W5**

Deliverables:

* LSTM baseline.
* Training pipeline.
* Initial metrics.

---

## M4 — Alert Prototype

**Cuối W7**

Deliverables:

* Fall detection.
* Event manager.
* Telegram notification.

---

## M5 — Integrated MVP

**Cuối W9**

Deliverables:

* AI + Backend + Frontend.
* Dashboard.
* Alert history.

---

## M6 — Final Testing

**Cuối W11**

Deliverables:

* Test report.
* Final metrics.
* Bug fixes.
* Final report draft.

---

## M7 — Final Product

**Cuối W12**

Deliverables:

* Source code.
* ≥20-page report.
* English slides.
* Demo video.
* Final presentation.

---

# 6. Phân công nhiệm vụ

## Thành viên 1 — AI/ML Engineer

### Trách nhiệm chính

* Dataset.
* MediaPipe.
* Feature engineering.
* LSTM/BiLSTM.
* Training.
* Evaluation.

### Deliverables

* Dataset pipeline.
* Training notebook/script.
* Model.
* Evaluation report.

---

# Thành viên 2 — Backend Engineer

### Trách nhiệm chính

* FastAPI.
* Model inference.
* Event management.
* Telegram Bot.
* Database/logging.
* Integration.

### Deliverables

* REST API.
* Inference service.
* Alert service.
* Event storage.

---

# Thành viên 3 — Frontend / QA / Documentation

### Trách nhiệm chính

* Streamlit UI.
* Dashboard.
* Testing.
* Report.
* Slides.
* Demo video.

### Deliverables

* Dashboard.
* Test cases.
* Test report.
* Final documentation.

---

# 7. Cách phối hợp nhóm

Repository:

```text
fallguard-ai/
│
├── ai/
├── backend/
├── frontend/
├── data/
├── models/
├── tests/
├── docs/
├── notebooks/
├── requirements.txt
└── README.md
```

Branch:

```text
main
├── develop
├── feature/pose
├── feature/model
├── feature/backend
└── feature/dashboard
```

Quy tắc:

* Không push trực tiếp code thử nghiệm vào `main`.
* Mỗi feature tạo branch.
* Commit có nội dung rõ ràng.
* Pull Request trước khi merge.
* Model và dataset lớn không đưa trực tiếp vào Git repository.

---

# 8. Kế hoạch kiểm thử

## 8.1. Unit Testing

Kiểm tra:

* Pose preprocessing.
* Normalization.
* Sequence generation.
* Model prediction.
* Confidence calculation.
* Alert formatting.

---

## 8.2. Model Testing

Test các tình huống:

| Scenario      | Expected |
| ------------- | -------- |
| Walking       | Normal   |
| Standing      | Normal   |
| Sitting       | Normal   |
| Lying         | Normal   |
| Bending       | Normal   |
| Forward fall  | Fall     |
| Backward fall | Fall     |
| Side fall     | Fall     |

---

## 8.3. Integration Testing

Kiểm tra:

```text
Camera
 ↓
MediaPipe
 ↓
Model
 ↓
FastAPI
 ↓
Event Manager
 ↓
Telegram
```

Mỗi thành phần phải hoạt động đúng khi kết hợp với thành phần tiếp theo.

---

# 9. Performance Testing

Các thông số cần đo:

* FPS.
* Inference latency.
* Alert latency.
* CPU usage.
* RAM usage.
* GPU usage nếu có.
* Model size.

Bảng kết quả dự kiến:

| Metric          |                 Target |
| --------------- | ---------------------: |
| FPS             |               ≥ 10 FPS |
| Model inference |               < 500 ms |
| Alert latency   |                  ≤ 5 s |
| RAM             | Theo giới hạn máy demo |
| False Positive  |     Càng thấp càng tốt |
| Recall          |         ≥ 90% mục tiêu |

Các target này sẽ được điều chỉnh sau khi có benchmark thực tế.

---

# 10. Risk Management

| Risk                   | Probability | Impact | Owner    | Mitigation         |
| ---------------------- | ----------- | ------ | -------- | ------------------ |
| Dataset thiếu          | M           | H      | AI       | Bổ sung dataset    |
| Model không đạt        | M           | H      | AI       | Baseline + tuning  |
| False positive         | H           | H      | AI       | Temporal filtering |
| Camera không ổn định   | M           | M      | Backend  | Video fallback     |
| API lỗi                | M           | M      | Backend  | Retry/logging      |
| UI lỗi                 | M           | M      | Frontend | Integration test   |
| Thiếu thời gian        | H           | H      | PM       | Feature freeze     |
| Thành viên quá tải     | M           | H      | PM       | Chia WBS rõ ràng   |
| Demo lỗi               | M           | H      | All      | Rehearsal          |
| Internet không ổn định | M           | M      | Backend  | Local inference    |

---

# 11. Communication Plan

| Hoạt động      | Tần suất       | Thời lượng | Nội dung          |
| -------------- | -------------- | ---------: | ----------------- |
| Daily check-in | 2–3 lần/tuần   | 10–15 phút | Tiến độ           |
| Team meeting   | 1 lần/tuần     | 30–60 phút | Sprint            |
| GVHD meeting   | 1 lần/1–2 tuần |    30 phút | Báo cáo           |
| Code review    | Khi merge      | 15–30 phút | Kiểm tra code     |
| Demo nội bộ    | Mỗi milestone  |    30 phút | Kiểm tra sản phẩm |

Mỗi thành viên nên báo cáo:

```text
Done:
- ...

Doing:
- ...

Blocked:
- ...
```

---

# 12. Sprint Planning

Dự án có thể sử dụng **6 Sprint × 2 tuần**.

| Sprint   | Thời gian | Mục tiêu              |
| -------- | --------- | --------------------- |
| Sprint 1 | W1–W2     | Requirement + dataset |
| Sprint 2 | W3–W4     | Pose + baseline       |
| Sprint 3 | W5–W6     | LSTM/BiLSTM           |
| Sprint 4 | W7–W8     | Backend + alert       |
| Sprint 5 | W9–W10    | Integration + testing |
| Sprint 6 | W11–W12   | Finalization          |

---

# 13. Ngân sách dự kiến

Dự án ưu tiên sử dụng công cụ miễn phí.

| Hạng mục         |                          Chi phí dự kiến |
| ---------------- | ---------------------------------------: |
| Python           |                                    0 VNĐ |
| OpenCV           |                                    0 VNĐ |
| MediaPipe        |                                    0 VNĐ |
| TensorFlow/Keras |                                    0 VNĐ |
| FastAPI          |                                    0 VNĐ |
| Streamlit        |                                    0 VNĐ |
| Git/GitHub       |                                    0 VNĐ |
| Google Colab     |                      0 VNĐ với free tier |
| Telegram Bot     |                                    0 VNĐ |
| Webcam           |                        Có sẵn / tùy nhóm |
| Hosting          |                           Không bắt buộc |
| **Tổng**         | **≈ 0 VNĐ nếu tận dụng thiết bị sẵn có** |

Có thể phát sinh chi phí nếu nhóm cần:

* Cloud GPU.
* SMS qua Twilio.
* Hosting.
* Camera bổ sung.

---

# 14. Acceptance Criteria

Sản phẩm được xem là hoàn thành MVP khi:

### AI

* [ ] MediaPipe trích xuất được pose.
* [ ] Sequence được tạo đúng.
* [ ] Model có thể phân loại Fall/Normal.
* [ ] Có evaluation trên test set.
* [ ] Có Accuracy, Precision, Recall, F1-score.
* [ ] Có confusion matrix.

### Backend

* [ ] API hoạt động.
* [ ] Model inference hoạt động.
* [ ] Fall event được ghi nhận.
* [ ] Telegram nhận được cảnh báo.

### Frontend

* [ ] Dashboard chạy được.
* [ ] Hiển thị trạng thái camera.
* [ ] Hiển thị prediction.
* [ ] Hiển thị confidence.
* [ ] Hiển thị lịch sử cảnh báo.

### Privacy

* [ ] Không lưu video gốc mặc định.
* [ ] Không sử dụng nhận diện khuôn mặt.
* [ ] Có chế độ skeleton visualization.

### Documentation

* [ ] Báo cáo ≥20 trang.
* [ ] Slide tiếng Anh.
* [ ] README.
* [ ] Source code.
* [ ] Demo video.

---

# 15. Definition of Done

Một task chỉ được xem là hoàn thành khi:

1. Code đã chạy.
2. Có kiểm thử cơ bản.
3. Không phá vỡ chức năng hiện tại.
4. Được commit vào repository.
5. Có tài liệu/hướng dẫn nếu cần.
6. Được thành viên khác kiểm tra.
7. Nếu liên quan AI, có kết quả thực nghiệm.

---

# 16. Chiến lược MVP

Để tránh tình trạng dự án bị quá lớn, nhóm nên **khóa phạm vi MVP** ở:

```text
Webcam
   ↓
MediaPipe Pose
   ↓
LSTM/BiLSTM
   ↓
Fall / Normal
   ↓
FastAPI
   ↓
Streamlit
   ↓
Telegram Alert
```

Không nên ưu tiên ngay:

```text
Mobile App
Multiple Cameras
SMS
Face Recognition
Cloud Deployment
Hospital Integration
Real-time Multi-user
```

Các tính năng này chỉ được thực hiện sau khi MVP đã ổn định.

---

# 17. Tiêu chí đánh giá sản phẩm cuối

Có thể chia thành 5 nhóm để chuẩn bị cho cả đồ án và cuộc thi:

| Nhóm           | Nội dung                        |
| -------------- | ------------------------------- |
| AI Performance | Accuracy, Precision, Recall, F1 |
| Technical      | Kiến trúc, model, latency       |
| Practicality   | Khả năng triển khai thực tế     |
| Privacy        | Không lưu video, skeleton-based |
| Demo           | Tính ổn định và dễ hiểu         |

Một điểm quan trọng trong phần trình bày là **không chỉ nói "model đạt accuracy cao"**. Nhóm nên chứng minh cả:

```text
Accuracy
+ Recall
+ Precision
+ F1
+ False Positive
+ False Negative
+ Latency
+ Privacy
```

Điều này giúp sản phẩm thể hiện rõ hơn tính chất của một **hệ thống AI hoàn chỉnh**, thay vì chỉ là một mô hình phân loại.

---

# 18. Ghi chú

Các nội dung cần bổ sung sau:

* [ ] Tên 3 thành viên.
* [ ] GVHD.
* [ ] Ngày bắt đầu chính xác.
* [ ] Deadline ĐACN2.
* [ ] Deadline AI4Life 2026.
* [ ] Dataset được chọn.
* [ ] Model architecture cuối cùng.
* [ ] Thiết bị demo.
* [ ] Kết quả benchmark.
* [ ] Chi phí thực tế.
* [ ] Link GitHub.
* [ ] Link demo.
* [ ] Kịch bản video demo.
* [ ] Tiêu chí chấm chính thức của AI4Life 2026.

---

## Kết luận định hướng

Với nhóm **2–3 sinh viên và 10–12 tuần**, tôi khuyến nghị coi **LSTM/BiLSTM + MediaPipe Pose + Telegram + Streamlit** là **core product**, còn React/mobile/multiple-camera/SMS chỉ là **extension**. Cách phân phạm vi này giúp nhóm vừa có đủ chiều sâu AI cho ĐACN2, vừa có một prototype hoàn chỉnh để trình diễn thay vì dành phần lớn thời gian xây dựng các thành phần phụ trợ.
