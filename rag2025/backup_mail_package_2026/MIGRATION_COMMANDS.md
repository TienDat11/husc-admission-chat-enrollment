# MIGRATION COMMANDS (Windows)

## A. Chuẩn bị
```bat
python --version
```

```bat
copy .env.example .env
```

## B. Cài deps
```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## C. Pipeline tự động
```bat
setup_data.bat data\raw
```

## D. Khởi chạy API
```bat
run_lab.bat
```

## E. Đánh giá nhanh
```bat
python scripts\evaluate_graphrag.py
```

## F. Chạy notebook (Jupyter)
```bat
python -m pip install jupyter
jupyter notebook
```

Mở lần lượt:
1. `notebooks/01_chunking_stats_for_thesis.ipynb`
2. `notebooks/02_graphrag_eval_for_thesis.ipynb`
3. `notebooks/03_latex_fill_helper.ipynb`

## G. Đóng gói backup lần cuối
```powershell
Compress-Archive -Path "backup_mail_package_2026\*" -DestinationPath "backup_mail_package_2026.zip" -Force
```
