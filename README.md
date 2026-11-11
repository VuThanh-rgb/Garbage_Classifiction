# Garbage_Classifiction

Đây là project phân loại rác sử dụng Deep Learning. Mục tiêu của project là xây dựng mô hình có khả năng nhận diện và phân loại rác thông qua hình ảnh đầu vào.

Project bao gồm quá trình xây dựng, huấn luyện và đánh giá mô hình trên tập dữ liệu hình ảnh, đồng thời triển khai ứng dụng để người dùng có thể thử nghiệm mô hình trực tiếp.

## Cấu trúc project

```text
Garbage-Classification/
│
├── App.py
├── Garbage Classification.ipynb
├── best_model_v3_1.pth
├── requirements.txt
├── README.md
└── .gitignore
```

## Các thành phần chính

- `App.py`: Chương trình giao diện demo, cho phép người dùng đưa hình ảnh vào để thực hiện phân loại.
- `Garbage Classification.ipynb`: Notebook chứa quá trình xử lý dữ liệu, xây dựng, huấn luyện và đánh giá mô hình.
- `best_model_v3_1.pth`: File trọng số của mô hình sau khi huấn luyện, được sử dụng trong ứng dụng demo.
- `requirements.txt`: Danh sách các thư viện cần thiết để chạy project.

## Cài đặt

Clone project về máy:

```bash
git clone https://github.com/VuThanh-rgb/Garbage-Classification.git
```

Di chuyển vào thư mục project:

```bash
cd Garbage-Classification
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

Sau khi cài đặt đầy đủ thư viện và các file cần thiết, chạy ứng dụng bằng lệnh:

```bash
python App.py
```

## Mô hình

Project sử dụng mô hình Deep Learning để thực hiện bài toán phân loại hình ảnh rác thải.

File `best_model_v3_1.pth` chứa trọng số đã được huấn luyện và sử dụng để dự đoán trên các hình ảnh đầu vào mà không cần huấn luyện lại từ đầu.

## Dataset

Dữ liệu sử dụng trong project là tập hình ảnh rác thải, phục vụ cho quá trình huấn luyện và đánh giá mô hình.

Do dung lượng dữ liệu có thể lớn, dataset không được đưa trực tiếp lên repository. Người dùng cần tải dữ liệu riêng và đặt đúng đường dẫn nếu muốn chạy lại notebook huấn luyện.

## Hướng phát triển

- Cải thiện độ chính xác của mô hình.
- Thử nghiệm thêm các phương pháp tiền xử lý và tăng cường dữ liệu.
- Mở rộng dữ liệu huấn luyện để cải thiện khả năng nhận diện.
- Hoàn thiện giao diện để việc sử dụng ứng dụng thuận tiện hơn.
