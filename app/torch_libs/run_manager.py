from pathlib import Path
from time import time

import torch
import polars as pl
from polars.exceptions import NoDataError


class RunManager:
    def __init__(self, exc_path=None, exp_name="exp_default", exp_path=None, run_id=None):
        """
        ex.)
            run = RunManager(exc_path=__file__, exp_name="exp_nyancat")
        """

        # pa_pathとexp_nameを指定 -> 格納ディレクトリとexp_nameを別々に指定
        # if pa_path is not None:
        #     pa_path = Path(pa_path).resolve()
        #     exp_path = pa_path / Path(exp_name)

        # exc_path(__file__)とexp_nameを指定 -> コード実行ディレクトリと同じディレクトリに結果を格納
        if exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)

        # exp_pathを指定 -> 格納ディレクトリとexp_nameを同時に指定
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        # 結果格納用のパスを設定し、適宜ディレクトリを作成
        exp_path.mkdir(parents=True, exist_ok=True)

        runs_path = exp_path / Path("runs")
        runs_path.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            run_id = self._get_run_id(runs_path)
        else:
            self._inherit_stats()

        run_path = runs_path / str(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.runs_path = runs_path
        self.run_path = run_path
        # self.run_id = run_id

        # self.start_time = time()

    # def __call__(self, fname):
    #     return self.run_path / Path(fname)

    def fpath(self, fname):
        return self.run_path / Path(fname)

    def _get_run_id(self, runs_path):
        dir_names = list(runs_path.iterdir())
        dir_nums = [int(dir_name.name) for dir_name in dir_names]
        if len(dir_nums) == 0:
            run_id = 0
        else:
            run_id = max(dir_nums) + 1
        return run_id

    def log_text(self, text, fname):
        with open(self.fpath(fname), "w") as fh:
            fh.write(text)

    # def log_df(self, df, fname):
    #     df.write_csv(self.run_path / Path(fname))
    #     # df.write_csv(self(fname))

    def log_df2csv(self, df, fname, *args, **kwargs):
        df.write_csv(self.fpath(fname), *args, **kwargs)
        # df.write_csv(run.run_path / Path("xxx"), ...)

    def log_torch_save(self, object, fname):
        torch.save(object, self.fpath(fname))

    def log_param(self, name, value):
        self.log_params({name: value})

    def log_params(self, stored_dict):
        stored_dict = {k: [v] for k, v in stored_dict.items()}
        if hasattr(self, "df_params"):
            self.df_params = self.df_params.hstack(pl.DataFrame(stored_dict))
        else:
            self.df_params = pl.DataFrame(stored_dict)

    def log_metric(self, name, value, step=None):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict, step=None):
        if hasattr(self, "df_metrics"):
            for name, value in stored_dict.items():
                if step is None:
                    step = self.df_metrics["step"].max() + 1
                if step in self.df_metrics["step"]:
                    self.df_metrics = self._df_set_elem(self.df_metrics, name, "step", step, value)
                else:
                    df_tmp = pl.DataFrame({"step": [step], name: [value]})
                    self.df_metrics = pl.concat([self.df_metrics, df_tmp], how="diagonal_relaxed")
        else:
            if step is None:
                step = 1
            stored_dict = {k: [v] for k, v in stored_dict.items()}
            self.df_metrics = pl.DataFrame({"step": [step], **stored_dict})

    def _df_set_elem(self, df, column, index_column_name, index, value):
        # indexは存在しなければならない。columnは存在しないとき新たに作られる。
        if value is not None:
            if column in df.columns:
                df = df.with_columns(pl.when(df[index_column_name] == index).then(value).otherwise(pl.col(column)).alias(column))
            else:
                df = df.with_columns(pl.when(df[index_column_name] == index).then(value).otherwise(pl.lit(None)).alias(column))
        return df
    
    def _inherit_stats(self):
        if self.fpath("param.csv").exists():
            self.df_params = pl.read_csv(self.fpath("param.csv"))
        if self.fpath("metrics.csv").exists():
            self.df_metrics = pl.read_csv(self.fpath("metrics.csv"))

    def _fetch_stats(self):
        if hasattr(self, "df_params"):
            stat_params = self.df_params
            if hasattr(self, "df_metrics"):
                stat_metrics = self.df_metrics[-1].select(pl.all().exclude("step"))
                stats = stat_params.hstack(stat_metrics)
            else:
                stats = stat_params
        else:
            if hasattr(self, "df_metrics"):
                stat_metrics = self.df_metrics[-1].select(pl.all().exclude("step"))
                stats = stat_metrics
            else:
                stats = pl.DataFrame()
        return stats

    def ref_stats(self, itv=None, step=None, last_step=None):
        if itv is None or (step - 1) % itv >= itv - 1 or step == last_step:
            if hasattr(self, "df_params"):
                self.df_params.write_csv(self.fpath("params.csv"))
            if hasattr(self, "df_metrics"):
                self.df_metrics.write_csv(self.fpath("metrics.csv"))
            self._fetch_stats().write_csv(self.fpath("stats.csv"))


class RunsManager:
    def __init__(self, runs):
        self.runs = runs

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, run in enumerate(self.runs):
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.runs) else arg for arg in args]
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(run, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper

    def __getitem__(self, idx):
        return self.runs[idx]

    # こいつらはこのままの形式だと格納できないから特別
    def log_param(self, name, value):
        self.log_params({name: value})

    def log_params(self, stored_dict):
        for i, run in enumerate(self.runs):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            run.log_params(new_stored_dict)

    def log_metric(self, name, value, step=None):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict, step=None):
        for i, run in enumerate(self.runs):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            run.log_metrics(new_stored_dict, step=step)


class RunViewer:
    def __init__(self, exc_path=None, exp_name="exp_default", exp_path=None):
    # def __init__(self, pa_path=None, exc_path=None, exp_name="exp_default", exp_path=None):
        # pa_pathとexp_nameを指定 -> 格納ディレクトリとexp_nameを別々に指定

        # if pa_path is not None:
            # pa_path = Path(pa_path).resolve()
            # exp_path = pa_path / Path(exp_name)

        # exc_path(__file__)とexp_nameを指定 -> コード実行ディレクトリと同じディレクトリに結果を格納
        if exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)

        # exp_pathを指定 -> 格納ディレクトリとexp_nameを同時に指定
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        runs_path = exp_path / Path("runs")

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.runs_path = runs_path

    def write_results(self, df, fname="results.csv"):
        df.write_csv(str(self.exp_path / Path(fname)))
        
    # def read_results(self, fname="results.csv", infer_schema_length=200):
    #     df = pl.read_csv(str(self.exp_path / Path(fname)), infer_schema_length=infer_schema_length)

    #     return df

    def fetch_results(self, fname="results.csv", refresh=False, met_listed=True):
        run_ids, params_paths = self.fetch_files("params.csv")
        run_ids, metrics_paths = self.fetch_files("metrics.csv")

        stats_l = []
        for run_id, params_path, metrics_path in zip(run_ids, params_paths, metrics_paths):
            df_run = pl.DataFrame({"run_id": run_id})
            try:
                df_params = pl.read_csv(params_path)
                df_run = df_run.hstack(df_params)
            except FileNotFoundError:
                pass
            try:
                df_metrics = pl.read_csv(metrics_path)
                df_metrics_list = df_metrics.select(pl.all().implode())
                df_run = df_run.hstack(df_metrics_list)
            except FileNotFoundError:
                pass
            # except NoDataError:
            #     print(f"No Data: {params_path}")
            #     pass
            
            stats_l.append(df_run)
            
        df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("run_id"))
        
        if refresh  or  not met_listed:
            df = df.with_columns(pl.col([col for col, dtype in df.schema.items() if dtype.is_nested()]).list.last().name.keep()) 

            if refresh:
                self.write_results(df, fname)

        return df

    # def fetch_results(self, fname="results.csv", refresh=True, met_listed=False):
    #     if refresh:
    #         try:
    #             df = self.ref_results(fname)
    #         except FileNotFoundError:
    #             df = self.read_results(fname)
    #     else:
    #         df = self.read_results(fname)

    #     return df

    def _fetch_metrics(self, listed=False):
        run_ids, stats_paths = self.fetch_files("metrics.csv")

        stats_l = []
        for run_id, stats_path in zip(run_ids, stats_paths):
            try:
                df_stats = pl.read_csv(stats_path)
                df_stats_wid = df_stats.with_columns(pl.lit(run_id).alias("run_id"))
                df_stats_wid = df_stats_wid.select(["run_id"] + df_stats.columns)
                stats_l.append(df_stats_wid)
            except FileNotFoundError:
                # metrics.csvがない場合Errorが発生する。
                pass
        if listed:
            return stats_l
        else:
            df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("step")).sort(pl.col("run_id"))
            return df
        
    def fetch_files(self, fname):
        dir_names = list(self.runs_path.iterdir())
        run_ids = [int(dir_name.name) for dir_name in dir_names]
        stats_paths = [dir_name / Path(fname) for dir_name in dir_names]
        
        return run_ids, stats_paths
        
