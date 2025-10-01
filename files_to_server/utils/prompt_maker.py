import importlib.util
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import tempfile
from pathlib import Path
# update dataset_helper when using new dataset
from dataset_helper import pubmedqa_helper, dreaddit_helper
PROMPTS_PY    = "prompt/prompts.py"
class promptMaker:
    def __init__(self, potential_dataset):
        self.potential_dataset = potential_dataset
        self.dataset2prompttemplate = {}
        self.path_to_prompt = 'prompt'
        self.templatehelper = {}
        self.helper_map = {
            # update when adding new dataset
            "pubmedqa": pubmedqa_helper,
            "dreaddit": dreaddit_helper,
        }
        self.pm = None
        self.setuppm()
        self.setuptemplateandhelper()
    
    def setuppm(self):
        here = Path(__file__).resolve()          # .../files_to_server/utils/prompt_maker.py
        prompts_py = here.parent / "prompts.py"  # .../files_to_server/utils/prompts.py

        if not prompts_py.exists():
            raise FileNotFoundError(f"prompts.py not found at: {prompts_py}")

        spec = importlib.util.spec_from_file_location("utils.prompts", prompts_py)
        prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts)

        # 依你的實作使用：有的是 class、有的是 factory，擇一
        self.pm = prompts.prompts_maker() 
    def setuptemplateandhelper(self):
        for pd in self.potential_dataset:
            with open(os.path.join(self.path_to_prompt, f"{pd}.txt"), "r", encoding='utf-8') as f:
                TEMPLATE_STR = f.read()
                self.templatehelper[pd] = getattr(self.helper_map[pd], pd + "_helper")(self.pm, TEMPLATE_STR)
    
    def processingdataset(self, TRAIN_CSV, VAL_CSV):
        def _norm_key(x) -> str:
            return str(x).strip().lower()

        def route_load_df_mixed_path(csv_path: str, helpers: dict) -> pd.DataFrame:
            """讀入混合 CSV，依 dataset 分組 -> 寫成臨時子 CSV -> helper.load_df(子路徑) -> 合回來"""
            raw = pd.read_csv(csv_path)
            if "dataset" not in raw.columns:
                raise KeyError(f"{csv_path} 缺少 'dataset' 欄位")
            # 保留原行序以便最後還原
            raw["_orig_idx"] = range(len(raw))
            raw["_ds_key"] = raw["dataset"].map(_norm_key)

            parts = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for key, g in raw.groupby("_ds_key", sort=False):
                    helper = helpers.get(key)
                    if helper is None:
                        raise KeyError(f"未知的 dataset: '{key}'. 可用: {list(helpers.keys())}")

                    # 為該 dataset 寫出子 CSV（包含 _orig_idx 以保序）
                    sub_path = os.path.join(tmpdir, f"{key}.csv")
                    g.to_csv(sub_path, index=False, encoding="utf-8")

                    # 交給對應 helper 讀取（它只吃路徑）
                    part = helper.load_df(sub_path)

                    # 期望 helper 不會刪掉我們加的 _orig_idx；若擔心可 assert 一下
                    if "_orig_idx" not in part.columns:
                        raise ValueError(f"{key}.load_df() 請保留 '_orig_idx' 欄位以便還原順序")
                    parts.append(part)

            # 合併並依原順序還原
            out = pd.concat(parts, axis=0).sort_values("_orig_idx")
            out = out.drop(columns=["_orig_idx", "_ds_key"], errors="ignore").reset_index(drop=True)
            return out

        def route_build_example(row, helpers: dict):
            key = _norm_key(row["dataset"])
            helper = helpers.get(key)
            if helper is None:
                raise KeyError(f"未知的 dataset: {key}. 可用: {list(helpers.keys())}")
            return pd.Series(helper.build_example(row))  # 回傳 dict -> 展開成多欄
        train_df = route_load_df_mixed_path(TRAIN_CSV, self.templatehelper)
        val_df   = route_load_df_mixed_path(VAL_CSV,   self.templatehelper)

        train_out = train_df.apply(lambda r: route_build_example(r, self.templatehelper),
                                axis=1, result_type="expand")
        val_out   = val_df.apply(lambda r: route_build_example(r, self.templatehelper),
                                axis=1, result_type="expand")

        ds = DatasetDict({
            "train":      Dataset.from_pandas(train_out, preserve_index=False),
            "validation": Dataset.from_pandas(val_out,   preserve_index=False),
        })
        return ds

if __name__ == '__main__':
    test = promptMaker(["dreaddit", "pubmedqa"])
    ds = test.processingdataset()