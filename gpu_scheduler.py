import os
import time
import fcntl
import itertools
import multiprocessing
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, Any, IO

import torch
import pynvml


def get_device(
    avoid_used: bool = False, 
    util_th: int = 10,
    avoid_locked: bool = False,
    lock_path: Path = Path("~/.gpu_locks"), 
    lock: bool = False, 
) -> torch.device | tuple[torch.device | None, IO | None] | None:
    """
    使用可能なdeviceを探索して返す.

    Args:
        avoid_used (bool): TrueならGPU使用率(Util)を見て空きを探す.
        util_th (int): avoid_used=True時のGPU使用率の閾値(%).
        avoid_locked (bool): TrueならロックされているGPUを避ける.
        lock_path (Path): ロックファイルの保存先ディレクトリ. デフォルトはホームディレクトリ下.
        lock (bool): Trueならロックファイルを作成・維持して (device, handle) を返す.

    Returns:
        lock=False: torch.device or None
        lock=True:  (torch.device, IO) or (None, None)
    """
    
    if not torch.cuda.is_available():
        warnings.warn("\n[Warning] CUDA is not available. Returning 'cpu'.\n", RuntimeWarning)
        device = torch.device("cpu")
        return (device, None) if lock else device

    pynvml.nvmlInit()
    
    try:
        # チルダ(~)を展開してから絶対パスへ変換
        lock_path = Path(lock_path).expanduser().resolve()
        
        if lock:
            try:
                lock_path.mkdir(parents=True, exist_ok=True)
                # 念のため権限を広げるが、ホームディレクトリ下なら所有者は自分なので通常エラーにならない
                os.chmod(lock_path, 0o777)
            except OSError:
                pass

        for i in range(torch.cuda.device_count()):
            lock_file = lock_path / f"gpu_{i}.lock"
            f: IO | None = None
            
            # 1. Lock Check
            if avoid_locked:
                try:
                    if lock:
                        f = open(lock_file, "a+")
                    elif lock_file.exists():
                        f = open(lock_file, "r")
                    
                    if f:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError):
                    if f:
                        f.close()
                    continue

            # 2. Usage Check
            if avoid_used:
                try:
                    props = torch.cuda.get_device_properties(i)
                    uuid_str = str(props.uuid)
                    if not uuid_str.startswith("GPU-"):
                        uuid_str = f"GPU-{uuid_str}"
                    h = pynvml.nvmlDeviceGetHandleByUUID(uuid_str.encode("utf-8"))
                    if pynvml.nvmlDeviceGetUtilizationRates(h).gpu > util_th:
                        if f:
                            fcntl.flock(f, fcntl.LOCK_UN)
                            f.close()
                        continue
                except pynvml.NVMLError as e:
                    warnings.warn(f"NVML check failed for GPU {i}: {e}", RuntimeWarning)
                    if f:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        f.close()
                    continue
                except Exception as e:
                    warnings.warn(f"GPU {i} check failed: {e}", RuntimeWarning)
                    if f:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        f.close()
                    continue

            # 3. Finalize
            device = torch.device(f"cuda:{i}")

            if lock:
                if f is None: 
                    try:
                        f = open(lock_file, "a+")
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (OSError, IOError):
                        if f:
                            f.close()
                        continue
                return (device, f)
            else:
                if f:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return device

    finally:
        pynvml.nvmlShutdown()

    return (None, None) if lock else None

def generate_task_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """設定辞書から直積タスクリストを生成する"""
    fixed = {}
    sweep = {}
    for k, v in config.items():
        if isinstance(v, list):
            sweep[k] = v
        else:
            fixed[k] = v

    if not sweep:
        return [deepcopy(fixed)]

    keys, values = zip(*sweep.items())
    tasks = []
    for combo in itertools.product(*values):
        new_cfg = deepcopy(fixed)
        new_cfg.update(dict(zip(keys, combo)))
        tasks.append(new_cfg)
    return tasks

def _worker_wrapper(task_func: Callable, cfg: dict, gpu_id: int):
    """[内部用] 環境変数をセットしてタスクを実行するラッパー"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    p_name = multiprocessing.current_process().name
    try:
        print(f"[{p_name}] Start Task on GPU {gpu_id}")
        task_func(cfg)
        print(f"[{p_name}] Done Task on GPU {gpu_id}")
    except Exception:
        print(f"[Error] Task failed on GPU {gpu_id}")
        traceback.print_exc()

def _spawn_worker(task_func: Callable, cfg: dict, gpu_id: int) -> multiprocessing.Process:
    """[ヘルパー] ワーカープロセスの生成と起動"""
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(
        target=_worker_wrapper, 
        args=(task_func, cfg, gpu_id),
        name=f"Worker-GPU{gpu_id}"
    )
    p.start()
    return p

def _cleanup_finished_processes(running_procs: dict[multiprocessing.Process, IO]):
    """[ヘルパー] 終了したプロセスのロック解放とリスト削除"""
    for p in list(running_procs.keys()):
        if not p.is_alive():
            p.join()
            lock_handle = running_procs[p]
            if lock_handle:
                try:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                    lock_handle.close()
                except IOError as e:
                    print(f"[Warn] Failed to close lock: {e}")
            del running_procs[p]


def parallel_run(
    task_func: Callable[[dict[str, Any]], None],
    config: dict[str, Any],
    max_tasks: int | None = None,
    check_interval: float = 1.0,
    avoid_used: bool = False,
    util_th: int = 10,
    lock_path: Path = Path("~/.gpu_locks"),
):
    """
    GPUリソースを管理しながらタスクを並列実行するスケジューラ.

    Args:
        task_func: 実行する関数 exp(cfg).
        config: 設定辞書. リストの値はSweep対象となる.
        max_tasks: 同時実行タスク数の上限.
        check_interval: ポーリング間隔(秒).
        avoid_used (bool): TrueならGPU使用率を見て空きを探す.
        util_th (int): GPU使用率の閾値(%).
        lock_path (Path): ロックファイルの保存先. デフォルトはホームディレクトリ下.
    """
    
    tasks = generate_task_grid(config)
    print(f"[Scheduler] Total tasks: {len(tasks)}")
    
    # { process_object: lock_file_handle }
    running_procs: dict[multiprocessing.Process, IO] = {}

    try:
        while tasks or running_procs:
            # 1. Cleanup
            _cleanup_finished_processes(running_procs)

            # 2. Spawn
            if tasks and (max_tasks is None or len(running_procs) < max_tasks):
                # ロック確保 (multirunは排他制御必須のため lock=True, avoid_locked=True で固定)
                res = get_device(
                    avoid_used=avoid_used,
                    util_th=util_th,
                    avoid_locked=True, # ロック前提なのでTrueで効率化
                    lock_path=lock_path,
                    lock=True
                )
                
                if res and res[0] is not None:
                    device, lock_handle = res
                    gpu_id = device.index
                    
                    next_cfg = tasks.pop(0)
                    
                    p = _spawn_worker(task_func, next_cfg, gpu_id)
                    running_procs[p] = lock_handle
                    
                    print(f"[Scheduler] Assigned task to GPU {gpu_id}. (Running: {len(running_procs)}, Remaining: {len(tasks)})")
                else:
                    time.sleep(check_interval)
            else:
                if running_procs:
                    time.sleep(check_interval)

        print("[Scheduler] All tasks completed.")

    except KeyboardInterrupt:
        print("\n[Scheduler] KeyboardInterrupt received. Terminating workers...")
        for p in list(running_procs.keys()):
            if p.is_alive():
                p.terminate() # 強制終了
                p.join()
                # ロック解放
                lock_handle = running_procs[p]
                if lock_handle:
                    try:
                        fcntl.flock(lock_handle, fcntl.LOCK_UN)
                        lock_handle.close()
                    except Exception:
                        pass
        print("[Scheduler] Shutdown complete.")
