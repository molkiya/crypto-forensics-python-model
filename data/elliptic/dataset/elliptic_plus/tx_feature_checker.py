import pandas as pd
import numpy as np

# Предположим, df_class_feature уже определён

for idx, val in enumerate(df_class_feature["class"]):
    if not isinstance(val, (int, float, np.integer, np.floating, bool)) or pd.isna(val):
        print(f"Первая неподходящая строка найдена на индексе {idx}: значение = {val} (тип: {type(val)})")
        break
else:
    print("Все значения можно безопасно преобразовать в тензор.")