# ViTextVQA Spatial Graph + OCR Retrieval

Pipeline hiện tại gồm:

- OCR theo từng bbox
- build `spatial graph`
- text embedding bằng `BAAI/bge-m3`
- image crop embedding bằng `openai/clip-vit-base-patch32`
- `text-first CSE`
- answer generation bằng:
  - Gemini
  - OpenAI-compatible VLM
  - Qwen2.5-VL-7B
  - Vintern chạy local qua Hugging Face

## 1. Môi trường CUDA trên Vast

Khuyến nghị:

- Python `3.10`
- CUDA `12.4`
- GPU có ít nhất:
  - `16GB VRAM` nếu chạy `Qwen2.5-VL-7B`
  - `24GB VRAM` nếu muốn chạy thoải mái hơn hoặc thử Vintern HF

Tạo env:

```bash
conda create -n vitextvqa python=3.10 -y
conda activate vitextvqa
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Nếu bạn muốn cài riêng PyTorch CUDA trước:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Lưu ý:

- `transformers` mới chặn một số đường load checkpoint khi `torch < 2.6` vì vấn đề bảo mật liên quan `torch.load`
- nếu bạn chạy trên Vast/Linux, nên dùng `torch==2.6.0` hoặc mới hơn

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2. Cấu trúc dữ liệu

Ảnh test:

```text
vitextvqa/ViTextVQA_images/st_images/<image_id>.jpg
```

OCR đã nhận text:

```text
outputs/OCR_img/<image_id>/ocr_results.jsonl
```

Graph offline sau khi build:

```text
outputs/graph_test/<image_id>/
  graph.json
  graph_enriched.json
  embeddings/
    text_embeddings.npy
    crop_embeddings.npy
  artifacts/
```

## 3. Build graph offline cho toàn bộ test

```bash
python scripts/build_test_graphs_offline.py --resume
```

Nếu muốn lưu overlay:

```bash
python scripts/build_test_graphs_offline.py --resume --save-visuals
```

Nếu muốn lưu crop node:

```bash
python scripts/build_test_graphs_offline.py --resume --save-node-crops
```

## 4. Query text-first CSE

```bash
python scripts/query_text_first_cse.py "góc trái phía bức hình nói về gì ?" --image-id 1003
```

Dùng graph test đã build sẵn:

```bash
python scripts/query_text_first_cse.py "thí chủ nên làm điều gì ?" --image-id 404 --graph-root outputs/graph_test
```

## 5. VLM full flow

### Gemini

`.env`

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

Chạy:

```bash
python scripts/model.py "góc trái phía bức hình nói về gì ?" --image-id 1003 --backend gemini
```

Hoặc với graph test:

```bash
python scripts/model.py "thí chủ nên làm điều gì ?" --image-id 404 --graph-root outputs/graph_test --backend gemini
```

### Chỉ ảnh + query

```bash
python scripts/model_only.py "thí chủ nối nữa thì sẽ như thế nào ?" --image-id 404 --backend gemini
```

### Qwen2.5-VL-7B

```bash
python scripts/model_qwen.py "góc trái phía bức hình nói về gì ?" --image-id 1003
```

Hoặc:

```bash
python scripts/model_qwen.py "thí chủ nên làm điều gì ?" --image-id 404 --graph-root outputs/graph_test
```

### Vintern local qua Hugging Face

```bash
python scripts/model_vintern_hf.py "góc trái phía bức hình nói về gì ?" --image-id 1003
```

Script này không qua OpenAI-compatible API. Nó:

- chạy CSE
- lấy crop node
- ghép `ảnh gốc + crop node` thành một ảnh composite
- feed trực tiếp vào `5CD-AI/Vintern-1B-v3_5`

## 6. OCR theo bbox bằng Vintern API

Nếu bạn vẫn muốn OCR crop-level qua endpoint local:

```bash
python scripts/recognize_bboxes_vintern.py
```

## 7. Ghi chú quan trọng

- `model.py` và `model_qwen.py` hiện feed:
  - ảnh gốc
  - crop từng node truy hồi
  - prompt chứa `top subgraph + node text + bbox`
- Nếu `crop_path` không được lưu sẵn, script sẽ tự cắt lại crop từ `crop_bbox/bbox`
- Batch offline hiện load embedding model một lần rồi tái sử dụng cho toàn bộ vòng lặp
- `Qwen2.5-VL-7B` và `Vintern HF` nặng hơn khá nhiều so với embedding pipeline
- Nếu Hugging Face báo rate limit, nên đặt thêm:

```env
HF_TOKEN=your_hf_token
```

## 8. File chính

- `scripts/build_test_graphs_offline.py`: build offline toàn bộ test
- `scripts/query_text_first_cse.py`: query `top-k seed + CSE`
- `scripts/model.py`: full flow với Gemini / OpenAI-compatible VLM
- `scripts/model_only.py`: chỉ ảnh + query
- `scripts/model_qwen.py`: full flow với Qwen2.5-VL-7B
- `scripts/model_vintern_hf.py`: full flow với Vintern HF
- `scripts/utils/prompts.py`: prompt dùng chung
