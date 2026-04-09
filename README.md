# Nhận Diện Cảm Xúc Giọng Nói

Hệ thống nhận diện cảm xúc từ âm thanh, hỗ trợ cả mô hình học máy cổ điển (Random Forest, SVM, Logistic Regression) và học sâu (MLP, CNN).

---

## Mục Lục

- [Cấu Trúc Dữ Liệu](#cấu-trúc-dữ-liệu)
- [Cài Đặt](#cài-đặt)
- [Cách Chạy](#cách-chạy)
- [Cấu Hình](#cấu-hình)
- [Các Pipeline Sẵn Có](#các-pipeline-sẵn-có)
- [Kết Quả](#kết-quả)

---

## Cấu Trúc Dữ Liệu

```
data/
├── data1/
│   ├── label.csv          # File nhãn chính
│   └── data/              # Thư mục chứa file âm thanh (.wav)
├── raw/                   # Dữ liệu tải về (lưu trữ tạm)
└── processed/             # Cache đặc trưng đã trích xuất
```

### Định dạng file `label.csv`

File CSV phải có ít nhất hai cột:

| Cột | Bắt buộc | Mô tả |
|-----|----------|-------|
| `path` | Có | Đường dẫn tới file `.wav` |
| `emotion` | Có | Nhãn cảm xúc (ví dụ: `happy`, `angry`, `sad`, `neutral`) |
| `speaker_id` | Không | Mã định danh người nói |
| `gender` | Không | Giới tính |
| `accent` | Không | Vùng miền / giọng địa phương |

**Ví dụ:**

```csv
index,speaker_id,path,duration,accent,emotion,emotion_id,gender
2736,0,visec/wavs/00000.wav,2.5,miền nam,happy,0,nữ
8475,1,visec/wavs/00001.wav,1.6,miền nam,neutral,1,nam
```

> **Lưu ý `path_mode`:** Cách đọc cột `path` phụ thuộc vào cài đặt `data.path_mode` trong config:
> - `"full"` — dùng đường dẫn nguyên vẹn trong CSV
> - `"relative"` — ghép với `data.audio_dir`
> - `"basename"` — chỉ lấy tên file, tìm trong `data.audio_dir`

### Định dạng file âm thanh

- **Định dạng:** `.wav`
- **Sample rate:** Tự động chuẩn hoá về 22.050 Hz
- **Thời lượng:** Tự động cắt hoặc đệm về 3 giây
- **Kênh:** Tự động chuyển sang mono

---

## Cài Đặt

**Yêu cầu:** Python 3.9+

```bash
pip install -r requirements.txt
```

**Cấu hình W&B (tuỳ chọn):**

```bash
cp .env.example .env
# Điền WANDB_API_KEY vào file .env nếu muốn dùng Weights & Biases
```

---

## Cách Chạy

### 1. Huấn luyện một pipeline

```bash
# Mặc định: MFCC + Random Forest
python train.py

# MFCC + MLP
python train.py feature_extraction=mfcc model=mlp pipeline_name=mfcc_mlp

# Mel-spectrogram + CNN
python train.py feature_extraction=melspec model=cnn pipeline_name=melspec_cnn

# Log-mel + CNN
python train.py feature_extraction=logmel model=cnn pipeline_name=logmel_cnn

# SVM với tham số tuỳ chỉnh
python train.py model=svm model.C=10.0 model.kernel=linear
```

### 2. Chạy toàn bộ pipeline

```bash
# Chạy cả 6 pipeline mặc định
python run_pipelines.py

# Bỏ qua một số pipeline (theo chỉ số 0-based)
python run_pipelines.py --skip 2,4
```

### 3. Tối ưu siêu tham số (Optuna)

```bash
# Tối ưu Random Forest (30 thử nghiệm)
python tune.py model=random_forest feature_extraction=mfcc tuning.n_trials=30

# Tối ưu MLP (20 thử nghiệm)
python tune.py model=mlp feature_extraction=mfcc tuning.n_trials=20

# Tối ưu CNN (15 thử nghiệm)
python tune.py model=cnn feature_extraction=melspec tuning.n_trials=15
```

Kết quả tốt nhất được lưu tại `outputs/best_params/{model}_{feature}.json`.

### 4. So sánh kết quả

```bash
# Xếp hạng theo F1 (mặc định)
python compare.py

# Xếp hạng theo độ chính xác
python compare.py --metric test/acc
```

Kết quả xuất ra `outputs/comparison.csv`.

---

## Cấu Hình

Dự án dùng [Hydra](https://hydra.cc/) để quản lý config. Mọi tham số đều có thể ghi đè trực tiếp trên dòng lệnh:

```bash
python train.py \
  feature_extraction=mfcc \
  model=mlp \
  training.lr=0.0001 \
  training.batch_size=16 \
  training.max_epochs=100 \
  data.train_ratio=0.8 \
  seed=123
```

**Các nhóm config chính:**

| Nhóm | Giá trị hợp lệ | Mô tả |
|------|---------------|-------|
| `feature_extraction` | `mfcc`, `melspec`, `logmel` | Phương pháp trích xuất đặc trưng |
| `model` | `random_forest`, `svm`, `logistic_regression`, `mlp`, `cnn` | Kiến trúc mô hình |
| `data.path_mode` | `full`, `relative`, `basename` | Cách đọc đường dẫn file âm thanh |
| `data.train_ratio` | 0.0–1.0 | Tỉ lệ dữ liệu train (mặc định 0.7) |
| `training.max_epochs` | số nguyên | Số epoch tối đa (chỉ áp dụng DL) |
| `logging.use_wandb` | `true`, `false` | Bật/tắt ghi log W&B |

---

## Các Pipeline Sẵn Có

| # | Tên | Đặc trưng | Mô hình | Đầu vào |
|---|-----|-----------|---------|---------|
| 0 | `mfcc_random_forest` | MFCC | Random Forest | `(80,)` |
| 1 | `mfcc_svm` | MFCC | SVM | `(80,)` |
| 2 | `mfcc_logistic_regression` | MFCC | Logistic Regression | `(80,)` |
| 3 | `mfcc_mlp` | MFCC | MLP | `(80,)` |
| 4 | `melspec_cnn` | Mel-spectrogram | CNN (ResNet18) | `(1, 128, T)` |
| 5 | `logmel_cnn` | Log-mel | CNN (ResNet18) | `(1, 128, T)` |

> MFCC cho đầu ra phẳng `(80,)` = 40 hệ số × 2 (mean + std).  
> Melspec/LogMel cho đầu ra 2D phù hợp với CNN.

---

## Kết Quả

Sau mỗi lần chạy, kết quả được lưu vào:

```
outputs/
├── runs/               # Kết quả từng pipeline (accuracy, F1, ...)
├── checkpoints/        # Checkpoint mô hình học sâu (.ckpt)
├── best_params/        # Siêu tham số tốt nhất từ Optuna (.json)
└── comparison.csv      # Bảng xếp hạng tổng hợp
artifacts/
└── models/             # Mô hình học máy cổ điển đã lưu (.pkl)
```
