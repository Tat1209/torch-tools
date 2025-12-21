import glob
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import polars as pl
from pl_utils import format_nested

from utils import interval, format_time

class RunPathManager:
    PARMAS_PQ = "_params.parquet"
    METRICS_PQ = "_metrics.parquet"
    RESULTS_PQ = "_results.parquet"
    
    PARAMS_CSV = "_params.csv"
    METRICS_CSV = "_metrics.csv"
    RESULTS_CSV = "_results.csv"

    def __init__(self, run_path: Path):
        self.run_path = run_path.resolve()
        self.run_id = self.run_path.name

    @property
    def params_path(self) -> Path:
        return self.run_path / self.PARMAS_PQ

    @property
    def metrics_path(self) -> Path:
        return self.run_path / self.METRICS_PQ

    def fpath(self, fname: str) -> Path:
        return self.run_path / fname


class RunManager(RunPathManager):
    def __init__(self, run_path: Path):
        super().__init__(run_path)
        self.df_params: Optional[pl.DataFrame] = None
        self.df_metrics: Optional[pl.DataFrame] = None
        
        self._params_buffer: Dict[str, Any] = {}
        self._metrics_buffer: List[Dict[str, Any]] = []

        self.run_path.mkdir(parents=True, exist_ok=True)
        self._inherit_stats()

    def log_text(self, fname: str, text: str):
        with open(self.fpath(fname), "w") as fh:
            fh.write(text)

    def log_torch_save(self, obj: Any, fname: str):
        import torch
        torch.save(obj, self.fpath(fname))
        
    def log_param(self, name: str, value: Any = None, delete: bool = False):
        """
        単一のパラメータを記録または削除する。
        valueがNoneの場合、何もしない（削除モードを除く）。
        """
        if delete:
            self.log_params([name], delete=True)
        elif value is not None:
            # Noneでない場合のみ実行
            self.log_params({name: value})

    def log_params(self, data: Union[Dict[str, Any], List[str]], delete: bool = False):
        """
        複数のパラメータを記録または削除する。
        辞書の値がNoneの項目はスキップされる。
        """
        # --- 削除モード ---
        if delete:
            if self.df_params is None:
                return
            
            keys = list(data.keys()) if isinstance(data, dict) else data
            existing = set(self.df_params.columns)
            targets = [k for k in keys if k in existing]
            
            if targets:
                self.df_params = self.df_params.drop(targets)
            return

        # --- 追加/上書きモード ---
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary for logging parameters.")

        processed_dict = {}
        for k, v in data.items():
            # 【変更点】値がNoneの場合はスキップして何もしない
            if v is None:
                continue

            try:
                if isinstance(v, list):
                    v = np.array(v)
                processed_dict[k] = [v]
            except Exception as e:
                raise ValueError(f"Failed to convert key '{k}' to NumPy array.") from e

        # 有効なデータが一つもなかった場合は終了
        if not processed_dict:
            return

        df_new = pl.DataFrame(processed_dict)

        if self.df_params is None:
            self.df_params = df_new
        else:
            existing_cols = set(self.df_params.columns)
            new_cols = set(df_new.columns)
            overlap = existing_cols.intersection(new_cols)

            if overlap:
                self.df_params = self.df_params.drop(list(overlap))
            
            self.df_params = self.df_params.hstack(df_new)

    def log_metric(self, name: str, value: Any, step: int):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict: Dict[str, Any], step: int):
        record = {"step": step, **stored_dict}
        self._metrics_buffer.append(record)

    def _flush(self):
        if self._params_buffer:
            df_new = pl.DataFrame(self._params_buffer)
            if self.df_params is None:
                self.df_params = df_new
            else:
                self.df_params = self.df_params.hstack(df_new)
            self._params_buffer = {}

        if self._metrics_buffer:
            df_new = pl.from_dicts(self._metrics_buffer, infer_schema_length=None)
            
            df_new = df_new.group_by("step", maintain_order=False).agg(
                pl.all().drop_nulls().last()
            ).sort("step")

            if self.df_metrics is None:
                self.df_metrics = df_new
            else:
                last_step = self.df_metrics["step"].item(-1)
                first_new_step = df_new["step"].item(0)

                if first_new_step > last_step:
                    self.df_metrics = pl.concat(
                        [self.df_metrics, df_new], 
                        how="diagonal_relaxed"
                    )
                else:
                    self.df_metrics = pl.concat(
                        [self.df_metrics, df_new], 
                        how="diagonal_relaxed"
                    ).group_by("step", maintain_order=False).agg(
                        pl.all().drop_nulls().last()
                    ).sort("step")
            
            self._metrics_buffer = []

    def _inherit_stats(self):
        if self.params_path.exists() and self.params_path.stat().st_size > 0:
            try:
                self.df_params = pl.read_parquet(self.params_path)
            except Exception:
                pass
        if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
            try:
                self.df_metrics = pl.read_parquet(self.metrics_path)
            except Exception:
                pass

    def sync(self, step=None, itv=None, last_step=None):
        if interval(step=step, itv=itv, last_step=last_step):
            self._flush()

            if self.df_params is not None:
                try:
                    self.df_params.write_parquet(self.params_path)
                    df_nonest = self.df_params.pipe(format_nested)
                    df_nonest.write_csv(self.fpath(self.PARAMS_CSV))
                except Exception as e:
                    print(f"Failed to write params: {e}")

            if self.df_metrics is not None:
                try:
                    self.df_metrics.write_parquet(self.metrics_path)
                    df_nonest = self.df_metrics.pipe(format_nested)
                    df_nonest.write_csv(self.fpath(self.METRICS_CSV))
                except Exception as e:
                    print(f"Failed to write metrics: {e}")


class RunLoader(RunPathManager):
    def load_params(self) -> Optional[pl.DataFrame]:
        if self.params_path.exists() and self.params_path.stat().st_size > 0:
            return pl.read_parquet(self.params_path)
        return None

    def load_metrics(self) -> Optional[pl.DataFrame]:
        if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
            return pl.read_parquet(self.metrics_path)
        return None

    def load_summary(self) -> pl.DataFrame:
        df = pl.DataFrame({"run_id": [self.run_id]})
        
        params = self.load_params()
        if params is not None:
            df = df.hstack(params)
            
        metrics = self.load_metrics()
        if metrics is not None:
            metrics_agg = metrics.select(pl.all().implode())
            df = df.hstack(metrics_agg)
            
        return df


class ExpManager:
    def __init__(self, exp_path: Union[str, Path], exp_tpl: str = "exp_tpl"):
        self.exp_path = Path(exp_path).resolve()
        self.runs_dir = self.exp_path / "runs"

        if not self.exp_path.exists():
            self._init_experiment(exp_tpl)
        
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _init_experiment(self, exp_tpl: str):
        self.exp_path.mkdir(parents=True)
        tpl_path = Path(__file__).resolve().parent / "exp_tpls" / exp_tpl
        if tpl_path.exists():
            for item in tpl_path.iterdir():
                dest = self.exp_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    def create_run(self, run_name_suffix: str = None) -> RunManager:
        run_id = f"run_{format_time(style=2)}"
        if run_name_suffix:
            run_id += f"_{run_name_suffix}"
        return RunManager(self.runs_dir / run_id)

    def list_runs(self) -> List[RunLoader]:
        if not self.runs_dir.exists():
            return []
        run_dirs = sorted([d for d in self.runs_dir.iterdir() if d.is_dir()])
        return [RunLoader(d) for d in run_dirs]

    def fetch_file_paths(self, fname: str, exist_only: bool = True) -> List[Path]:
        paths = []
        for loader in self.list_runs():
            target_path = loader.fpath(fname)
            if not exist_only or target_path.exists():
                paths.append(target_path)
        return paths

    def ref_results(self) -> pl.DataFrame:
        loaders = self.list_runs()
        df_list = []
        for loader in loaders:
            try:
                df_run = loader.load_summary()
                df_list.append(df_run)
            except Exception as e:
                print(f"Failed to load run {loader.run_id}: {e}")

        if not df_list:
            return pl.DataFrame()

        df_all = pl.concat(df_list, how="diagonal_relaxed").sort("run_id")
        
        try:
            df_all.write_parquet(self.exp_path / RunPathManager.RESULTS_PQ)
            df_nonest = df_all.pipe(format_nested)
            df_nonest.write_csv(self.exp_path / RunPathManager.RESULTS_CSV)
        except Exception as e:
            print(f"Failed to save aggregated results: {e}")
            
        return df_all
    
    def get_run(self, run_id: str) -> 'RunManager':
        """指定されたrun_idのRunManagerインスタンスを返す"""
        run_path = self.runs_dir / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found in {self.runs_dir}")
        return RunManager(run_path)
    
    def fetch_results(self, refresh: bool = False, run_id: bool = True, run_path: bool = False) -> pl.DataFrame:
        """
        Args:
            refresh: キャッシュを無視して再集計するか
            run_id: run_idカラムを含めるか (Default: True)
            run_path: 絶対パス(run_path)カラムを含めるか (Default: False)
        """
        results_pq_path = self.exp_path / RunPathManager.RESULTS_PQ
        
        if not refresh and results_pq_path.exists():
            try:
                df = pl.read_parquet(results_pq_path)
            except Exception:
                df = self.ref_results()
        else:
            df = self.ref_results()

        if df.is_empty():
            return df

        # run_path の追加
        if run_path:
            base_dir = str(self.runs_dir) + "/"
            df = df.with_columns(
                (pl.lit(base_dir) + pl.col("run_id")).alias("run_path")
            )

        if not run_id:
            df = df.drop("run_id")
        
        return df

class RunsManager:
    def __init__(self, runs: List[RunManager]):
        self.runs = runs
    
    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            n = len(self.runs)
            new_args = [[] for _ in range(n)]
            new_kwargs = [{} for _ in range(n)]

            for arg in args:
                if isinstance(arg, list) and len(arg) == n:
                    for i in range(n):
                        new_args[i].append(arg[i])
                elif isinstance(arg, dict):
                    per_dicts = [ {} for _ in range(n) ]
                    for k, v in arg.items():
                        pairable = isinstance(v, list) and len(v) == n
                        for i in range(n):
                            per_dicts[i][k] = v[i] if pairable else v
                    for i in range(n):
                        new_args[i].append(per_dicts[i])
                else:
                    for i in range(n):
                        new_args[i].append(arg)

            for kw, val in kwargs.items():
                if isinstance(val, list) and len(val) == n:
                    for i in range(n):
                        new_kwargs[i][kw] = val[i]
                elif isinstance(val, dict):
                    per_dicts = [{} for _ in range(n)]
                    for k, v in val.items():
                        pairable = isinstance(v, list) and len(v) == n
                        for i in range(n):
                            per_dicts[i][k] = v[i] if pairable else v
                    for i in range(n):
                        new_kwargs[i][kw] = per_dicts[i]
                else:
                    for i in range(n):
                        new_kwargs[i][kw] = val

            results = []
            for run, a, kw in zip(self.runs, new_args, new_kwargs):
                method = getattr(run, attr)
                results.append(method(*a, **kw))
            return results
        return wrapper

    def __getitem__(self, idx):
        return self.runs[idx]


def cat_results(exp_paths: List[Union[str, Path]], refresh: bool = False) -> pl.DataFrame:
    file_paths = set()
    for pattern in exp_paths:
        file_paths.update(glob.glob(str(pattern), recursive=True))

    dfs = []
    for file_path_str in sorted(list(file_paths)):
        exp_path = Path(file_path_str)
        if exp_path.is_dir():
            try:
                exp_name = exp_path.name
                viewer = ExpManager(exp_path=exp_path)
                
                df = viewer.fetch_results(refresh=refresh)
                if not df.is_empty():
                    df = df.with_columns(pl.lit(exp_name).alias("exp_name"))
                    df = df.select(["exp_name", *[c for c in df.columns if c != "exp_name"]])
                    dfs.append(df)
            except Exception as e:
                print(f"Skipping {exp_path}: {e}")
            
    if not dfs:
        return pl.DataFrame()
        
    return pl.concat(dfs, how="diagonal_relaxed")