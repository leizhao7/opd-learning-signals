"""prime 代码奖励的隔离打分子进程（2026-09-23 补丁 v2：常驻 worker 版）。

由 prime.py 的 run_reward_scoring 以独立会话（start_new_session）启动：
    python -u _prime_isolated_runner.py <in.pkl> <out.pkl>

语义与原 ProcessPoolExecutor 版本一致：
  - num_processes 个常驻 worker 并发，按输入顺序（FIFO）派发；
  - 整批一个截止时间（原代码 asyncio.gather 让所有 wait_for 同时起算，等价于"整批 timeout 秒"），
    截止时仍在跑或尚未开始的样本记为 None（上层换算为 0.0）；
  - 分数换算：None -> None；int/float/bool -> float；其余取 float(res[0])；任何异常 -> None。
与原版的差别只在进程管理：
  - worker 由本进程（单一控制流）一次性 fork，数据随 fork 继承，任务只下发一个下标；
    每个 worker 有私有的一对管道、自成进程组，不存在共享队列/锁；
  - 截止时按进程组整组 SIGKILL（连同评测代码派生的孙进程），本进程为 PR_SET_CHILD_SUBREAPER，
    过继来的孤儿由本进程 waitpid 回收，不在容器 1 号进程下堆僵尸；
  - worker 中途意外退出：该样本记 None，立即补一个新 worker。
截止时间从本进程就绪（import 打分模块、fork 完 worker）时起算，对应原版 worker 从已加载模块的父进程 fork。
"""
import json
import os
import pickle
import select
import signal
import struct
import sys
import time

for _p in reversed(json.loads(os.environ.get("PRIME_ISO_SYS_PATH", "[]"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HDR = struct.Struct("!I")


def _set_subreaper():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
        return True
    except Exception:
        return False


def _reap_all():
    n = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return n
        if pid == 0:
            return n
        n += 1


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _worker(func, tasks, comps, refs, extras, rfd, wfd):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    while True:
        hdr = _read_exact(rfd, 4)
        if hdr is None:
            os._exit(0)
        i = _HDR.unpack(hdr)[0]
        try:
            res = func(tasks[i], comps[i], refs[i], extras[i])
            if res is None:
                payload = ("none", None)
            elif isinstance(res, int | float | bool):
                payload = ("ok", float(res))
            else:
                payload = ("ok", float(res[0]))
        except BaseException as e:  # noqa: BLE001  与原版一样：任何异常都记 None
            payload = ("err", repr(e)[:500])
        data = pickle.dumps((i, payload))
        _write_all(wfd, _HDR.pack(len(data)) + data)


class _W:
    __slots__ = ("pid", "to_w", "from_w", "busy", "buf")


def _spawn(job_args):
    to_r, to_w = os.pipe()
    from_r, from_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(to_w)
        os.close(from_r)
        try:
            _worker(*job_args, to_r, from_w)
        finally:
            os._exit(0)
    os.close(to_r)
    os.close(from_w)
    try:
        os.setpgid(pid, pid)
    except OSError:
        pass
    w = _W()
    w.pid, w.to_w, w.from_w, w.busy, w.buf = pid, to_w, from_r, None, bytearray()
    return w


def _close(w):
    for fd in (w.to_w, w.from_w):
        try:
            os.close(fd)
        except OSError:
            pass


def main():
    inp, out = sys.argv[1], sys.argv[2]
    subreaper = _set_subreaper()
    with open(inp, "rb") as f:
        job = pickle.load(f)  # 反序列化 func 时 import 打分模块（只做一次）
    func = job["func"]
    comps, refs, tasks, extras = job["completions"], job["references"], job["tasks"], job["extra_info"]
    nproc, timeout = int(job["num_processes"]), float(job["timeout"])
    n = len(tasks)
    job_args = (func, tasks, comps, refs, extras)
    results = [None] * n
    status = ["pending"] * n
    workers = [_spawn(job_args) for _ in range(min(nproc, n))] if n else []
    t_ready = time.monotonic()
    deadline = t_ready + timeout
    queue = list(range(n - 1, -1, -1))  # pop() 取队首 -> FIFO
    n_err = respawned = 0

    def finish(i, kind, val):
        nonlocal n_err
        if kind == "ok":
            results[i], status[i] = val, "done"
        elif kind == "none":
            results[i], status[i] = None, "done"
        else:
            results[i], status[i] = None, "error"
            n_err += 1
            print(f"[Error] Task failed: {val}, completion: {str(comps[i])[:80]}", flush=True)

    while (queue or any(w.busy is not None for w in workers)) and time.monotonic() < deadline:
        for w in workers:
            if w.busy is None and queue:
                i = queue.pop()
                try:
                    _write_all(w.to_w, _HDR.pack(i))
                    w.busy = i
                except OSError:
                    queue.append(i)
        busy = {w.from_w: w for w in workers if w.busy is not None}
        wait = max(0.0, min(1.0, deadline - time.monotonic()))
        ready = select.select(list(busy), [], [], wait)[0] if busy else []
        for fd in ready:
            w = busy[fd]
            chunk = os.read(fd, 1 << 16)
            if not chunk:  # worker 意外退出：该样本记 None，补一个新 worker
                finish(w.busy, "err", "worker exited unexpectedly")
                _close(w)
                try:
                    os.killpg(w.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                workers[workers.index(w)] = _spawn(job_args)
                respawned += 1
                continue
            w.buf.extend(chunk)
            while len(w.buf) >= 4:
                (ln,) = _HDR.unpack(w.buf[:4])
                if len(w.buf) < 4 + ln:
                    break
                i, (kind, val) = pickle.loads(bytes(w.buf[4:4 + ln]))
                del w.buf[:4 + ln]
                finish(i, kind, val)
                w.busy = None
        _reap_all()
    killed_running = sum(1 for w in workers if w.busy is not None)
    for w in workers:
        if w.busy is not None:
            status[w.busy] = "timeout"
        _close(w)
        try:
            os.killpg(w.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for i in queue:
        status[i] = "timeout"
    n_to = 0
    for i in range(n):
        if status[i] == "timeout":
            n_to += 1
            print(f"[Timeout] Task timeout: {str(comps[i])[:200]}", flush=True)
    t_reap = time.monotonic() + 30
    while time.monotonic() < t_reap:
        _reap_all()
        try:
            os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        time.sleep(0.05)
    summary = dict(n=n, done=status.count("done"), error=n_err, timeout=n_to, killed_running=killed_running,
                   never_started=len(queue), respawned=respawned, wall_s=round(time.monotonic() - t_ready, 1),
                   subreaper=subreaper)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(results=results, summary=summary), f)
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
