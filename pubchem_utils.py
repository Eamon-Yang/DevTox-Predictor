"""
共享工具: 调用 PaDEL-Descriptor 计算 PubChem 分子指纹 (881位)。
被 build_preprocessor.py (重建预处理流水线) 和 app.py (网页预测) 共同调用,
确保两边算指纹的方式完全一致 —— 这是保证预测准确的关键。
"""
import os
import uuid
import pandas as pd
from padelpy import padeldescriptor


def compute_raw_pubchem_fingerprint(smiles_list):
    """
    输入: SMILES字符串的列表
    输出: DataFrame, 列名为 PubChem_0 ~ PubChem_880 (共881列),
          行顺序与输入的 smiles_list 顺序一致。

    注意: 这里假设传入的 SMILES 都是合法的 (调用方应先用 RDKit 校验过)。
    若 PaDEL 处理某个分子失败导致行数变化, 会在外层 app.py 中通过行数核对发现问题。
    """
    if len(smiles_list) == 0:
        return pd.DataFrame()

    uid = uuid.uuid4().hex[:8]
    temp_smi = f"temp_{uid}.smi"
    temp_csv = f"temp_out_{uid}.csv"

    try:
        with open(temp_smi, 'w') as f:
            f.write('\n'.join(str(s) for s in smiles_list) + '\n')

        # 与当年训练时完全一致的调用参数 —— 这几个参数任何一个改动都会导致指纹不一致
        padeldescriptor(
            mol_dir=temp_smi,
            d_file=temp_csv,
            fingerprints=True,
            detectaromaticity=True,
            standardizenitro=True,
            removesalt=True,
            log=False
        )

        df_fp = pd.read_csv(temp_csv)
        fp_cols = [c for c in df_fp.columns if 'PubchemFP' in c]
        df_fp = df_fp[fp_cols] if fp_cols else df_fp.drop(columns=['Name'], errors='ignore').iloc[:, :881]
        df_fp.columns = [f'PubChem_{i}' for i in range(df_fp.shape[1])]
        df_fp = df_fp.reset_index(drop=True)
        return df_fp

    finally:
        for f in [temp_smi, temp_csv]:
            if os.path.exists(f):
                os.remove(f)
