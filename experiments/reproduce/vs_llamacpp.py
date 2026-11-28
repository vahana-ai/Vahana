"""Reproduce Vahana vs llama.cpp on your own machine, with one command.

  git clone https://github.com/vahana-ai/Vahana && pip install "vahana[gpu]"     (or plain `vahana` without a GPU)
  python Vahana/experiments/reproduce/vs_llamacpp.py [--dir DIR] [--yes] [--quality] [--model mixtral-8x7b]

What it does, all on this machine:
  1. Downloads, after asking (--yes skips the question; files already present are reused):
       Vahana's model       RedHatAI/Qwen3-30B-A3B-quantized.w4a16     ~16.5 GB
       llama.cpp's model    Qwen/Qwen3-30B-A3B-GGUF, Q4_K_M            ~18.6 GB
       (--model mixtral-8x7b: Mixtral-8x7B-Instruct, 4-bit ~24.5 GB and Q4_K_M ~28.4 GB)
       llama.cpp            release b12494 for this OS / GPU           ~25-600 MB
       with --quality: the bf16 original Qwen/Qwen3-30B-A3B (~61 GB) and the WikiText-2 test text (5 MB)
  2. Tunes llama.cpp to its fastest setting here: how many MoE layers stay on the CPU (--n-cpu-moe), by a sweep
     with llama-bench. Vahana uses its defaults.
  3. Runs the head-to-heads (engines alternate, both warm; see experiments/v15_decode/README.md):
       benchmark text   decode tok/s after a 512-token prompt          (v15_decode/head_to_head.py)
       real prompts     five prompts, 96 greedy tokens: decode tok/s and time to first token
                                                                         (v15_decode/real_prompts_h2h.py)
     and with --quality, how close each 4-bit model is to the bf16 original on the same tokens: KL divergence,
     same top token, perplexity (v15_quality/kld.py; llama.cpp scores itself with llama-perplexity).
  4. Prints a table and writes everything to DIR/results/ (JSON and logs), with this machine's description.
"""
import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
LLAMA_TAG = "b12494"
# --model presets: Vahana's 4-bit checkpoint, llama.cpp's Q4_K_M of the same model, the bf16 original, MoE layers
MODELS = {
    "qwen3-30b": dict(vahana="RedHatAI/Qwen3-30B-A3B-quantized.w4a16", vgb=16.5, vdir="Qwen3-30B-A3B-w4a16",
                      gguf=("Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"), ggb=18.6,
                      bf16="Qwen/Qwen3-30B-A3B", bgb=61, layers=48),
    "mixtral-8x7b": dict(vahana="nm-testing/Mixtral-8x7B-Instruct-v0.1-W4A16-quantized", vgb=24.5,
                         vdir="Mixtral-8x7B-Instruct-w4a16",
                         gguf=("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF", "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"),
                         ggb=28.4, bf16="mistralai/Mixtral-8x7B-Instruct-v0.1", bgb=93, layers=32),
}
WIKI = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--dir", default=os.path.join(os.path.expanduser("~"), "vahana_vs_llamacpp"))
ap.add_argument("--yes", action="store_true", help="download without asking")
ap.add_argument("--quality", action="store_true", help="also compare quality against the bf16 original (~61 GB more for qwen3-30b)")
ap.add_argument("--llama-dir", default=None, help="use this llama.cpp build instead of downloading one")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--chunks", type=int, default=32, help="--quality: 512-token chunks of WikiText-2")
ap.add_argument("--copy-results", default=None, help="also copy results/ here when done (e.g. /kaggle/working)")
ap.add_argument("--model", default="qwen3-30b", choices=sorted(MODELS), help="which MoE model to compare on")
a = ap.parse_args()
MD = MODELS[a.model]
VAHANA_REPO, (GGUF_REPO, GGUF_FILE), BF16_REPO, N_LAYERS = MD["vahana"], MD["gguf"], MD["bf16"], MD["layers"]
os.makedirs(a.dir, exist_ok=True)
RES = os.path.join(a.dir, "results")
os.makedirs(RES, exist_ok=True)


def run(cmd, log=None, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if log:
        open(os.path.join(RES, log), "w", encoding="utf-8").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r


def machine():
    m = {"os": f"{platform.system()} {platform.release()}", "cpu": platform.processor() or platform.machine(),
         "logical_cpus": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        import psutil
        m["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        m["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            m["gpu"], m["gpu_gb"] = p.name, round(p.total_memory / 2**30, 1)
        m["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import vahana
        m["vahana"] = vahana.__version__
    except ImportError:
        sys.exit("vahana is not installed: pip install \"vahana[gpu]\" (or vahana)")
    return m


M = machine()
GPU = "gpu" in M
SYSTEM = platform.system()


def llama_assets():
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if SYSTEM == "Windows":
        if GPU:
            return [f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
        return [f"llama-{LLAMA_TAG}-bin-win-cpu-{arch}.zip"]
    if SYSTEM == "Darwin":
        return [f"llama-{LLAMA_TAG}-bin-macos-{arch}.tar.gz"]
    if GPU:
        return [f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"]
    return [f"llama-{LLAMA_TAG}-bin-ubuntu-{arch}.tar.gz"]


def find_exe(root, name):
    exe = name + (".exe" if SYSTEM == "Windows" else "")
    for d, _, fs in os.walk(root):
        if exe in fs:
            return d
    return None


# ---------------------------------------------------------------- 1. downloads
vdir = os.path.join(a.dir, MD["vdir"])
gguf = os.path.join(a.dir, GGUF_FILE)
bdir = os.path.join(a.dir, BF16_REPO.split("/")[-1])
ldir = a.llama_dir or os.path.join(a.dir, f"llama.cpp-{LLAMA_TAG}")
todo = []


def complete(d):
    """Every weight file the checkpoint's index names is present (a stopped download leaves config.json alone)."""
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return os.path.exists(os.path.join(d, "model.safetensors"))
    return all(os.path.exists(os.path.join(d, f)) for f in set(json.load(open(idx))["weight_map"].values()))


if not complete(vdir):
    todo.append(f"Vahana's model {VAHANA_REPO} (~{MD['vgb']} GB)")
if not os.path.exists(gguf):
    todo.append(f"llama.cpp's model {GGUF_REPO}/{GGUF_FILE} (~{MD['ggb']} GB)")
if not a.llama_dir and not find_exe(ldir, "llama-bench"):
    todo.append(f"llama.cpp {LLAMA_TAG}: " + ", ".join(llama_assets()))
if a.quality and not complete(bdir):
    todo.append(f"the bf16 original {BF16_REPO} (~{MD['bgb']} GB) and WikiText-2 (5 MB)")
print("machine:", json.dumps(M))
if todo:
    print(f"\nTo download into {a.dir}:\n  - " + "\n  - ".join(todo))
    if not a.yes and input("Download these? [y/N] ").strip().lower() != "y":
        sys.exit("nothing downloaded")
    from huggingface_hub import hf_hub_download, snapshot_download
    if not complete(vdir):
        snapshot_download(VAHANA_REPO, local_dir=vdir)
    if not os.path.exists(gguf):
        hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=a.dir)
    if a.quality and not complete(bdir):
        snapshot_download(BF16_REPO, local_dir=bdir)
    if not a.llama_dir and not find_exe(ldir, "llama-bench"):
        os.makedirs(ldir, exist_ok=True)
        for asset in llama_assets():
            f = os.path.join(a.dir, asset)
            if not os.path.exists(f):
                url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}"
                print("  downloading", url, flush=True)
                urllib.request.urlretrieve(url, f + ".part")
                os.replace(f + ".part", f)
            if f.endswith(".zip"):
                zipfile.ZipFile(f).extractall(ldir)
            else:
                with tarfile.open(f) as t:
                    t.extractall(ldir, filter="data")
        if SYSTEM == "Windows" and GPU:                 # the CUDA runtime DLLs go next to the executables
            exe_dir = find_exe(ldir, "llama-bench")
            for x in os.listdir(ldir):
                if x.lower().endswith(".dll") and exe_dir != ldir:
                    shutil.copy(os.path.join(ldir, x), exe_dir)
lbin = find_exe(ldir, "llama-bench")
if not lbin:
    sys.exit(f"no llama-bench under {ldir}")
if SYSTEM != "Windows":
    for x in os.listdir(lbin):
        p = os.path.join(lbin, x)
        if os.path.isfile(p) and x.startswith("llama-"):
            os.chmod(p, 0o755)
env_lib = dict(os.environ)
if SYSTEM == "Linux":                                   # llama.cpp's libraries and the CUDA runtime it ships
    libs = sorted({d for d, _, fs in os.walk(ldir) if any(".so" in f for f in fs)} | {lbin})
    env_lib["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env_lib.get("LD_LIBRARY_PATH", "")])
os.environ.update(env_lib)

# ---------------------------------------------------------------- 2. tune llama.cpp
def bench(n, t, timeout=900):
    """llama-bench decode tok/s (median of the warm repetitions) at --n-cpu-moe n (None: CPU only) and -t t, or None
    if it did not run or ran out of time."""
    gpu_args = ["-ngl", "99", "--n-cpu-moe", n] if n is not None else ["-ngl", "0"]
    try:
        r = run([os.path.join(lbin, "llama-bench"), "-m", gguf, *gpu_args, "-t", t, "-p", "0", "-n", "32", "-r", "3",
                 "-o", "jsonl"], timeout=timeout)
        return statistics.median(json.loads(r.stdout.strip().splitlines()[0])["samples_ts"][1:])
    except (subprocess.TimeoutExpired, IndexError, ValueError, KeyError):
        return None


threads = str(M.get("physical_cores") or max(1, (os.cpu_count() or 2) // 2))
ngl = "99" if GPU or SYSTEM == "Darwin" else "0"
best = ("0", None)
if ngl != "0" and SYSTEM != "Darwin":
    print(f"\ntuning llama.cpp: --n-cpu-moe sweep (-t {threads})", flush=True)
    # From the most layers on the GPU to the fewest: settings that do not fit fail fast at load, the first that fits
    # is usually near the best, and the sweep stops once a setting is slower than the best so far. (Starting from
    # every layer on the CPU, a model larger than RAM can page for an hour per setting.) A setting that runs out of
    # time counts as failed.
    for n in sorted({v for v in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44) if v < N_LAYERS} | {N_LAYERS}):
        ts = bench(n, threads)
        if ts is None:
            print(f"    --n-cpu-moe {n}: did not run (GPU memory or time)", flush=True)
            if best[1] is None:
                continue
            break
        print(f"    --n-cpu-moe {n}: {ts:.2f} tok/s", flush=True)
        if best[1] is not None and ts < best[1]:
            break
        best = (str(n), ts)
    if best[1] is None:
        sys.exit("llama.cpp did not run with any --n-cpu-moe setting")
    logical = str(os.cpu_count() or threads)
    if logical != threads:                               # llama.cpp may prefer every logical CPU
        ts = bench(best[0], logical)
        if ts is not None:
            print(f"    --n-cpu-moe {best[0]} -t {logical}: {ts:.2f} tok/s", flush=True)
            if ts > best[1]:
                threads, best = logical, (best[0], ts)
    print(f"  llama.cpp's best: --n-cpu-moe {best[0]} -t {threads}", flush=True)
if ngl == "0":                                          # CPU only: llama.cpp's faster thread count
    print("\ntuning llama.cpp: threads (CPU only)", flush=True)
    tbest = None
    for t in sorted({threads, str(os.cpu_count() or threads)}, key=int):
        ts = bench(None, t, timeout=3600)
        if ts is None:
            continue
        print(f"    -t {t}: {ts:.2f} tok/s", flush=True)
        if tbest is None or ts > tbest[1]:
            tbest = (t, ts)
    if tbest:
        threads = tbest[0]
    print(f"  llama.cpp's best: -t {threads}", flush=True)
    # Vahana's default (every CPU but 2) is set for GPU machines, where a CPU feeds the GPU; CPU-only, tune it too
    print("\ntuning Vahana: threads (CPU only)", flush=True)
    n = os.cpu_count() or 2
    vbest = None
    for t in sorted({max(1, n - 2), int(M.get("physical_cores") or 1), n}):
        r = run([sys.executable, os.path.join(EXP, "v15_decode", "profile_decode.py"), vdir], timeout=3600,
                env=dict(os.environ, VAHANA_THREADS=str(t), PROFILE_DECODE_ONLY="1", OMP_WAIT_POLICY="PASSIVE"))
        line = next((x for x in r.stdout.splitlines() if x.startswith("decode:")), None)
        if line is None:
            print(f"    {t} threads: failed\n{r.stderr[-3000:]}", flush=True)
            continue
        ts = float(line.split()[1])
        print(f"    {t} threads: {ts:.2f} tok/s", flush=True)
        if vbest is None or ts > vbest[1]:
            vbest = (t, ts)
    if vbest:
        os.environ["VAHANA_THREADS"] = str(vbest[0])         # inherited by both head-to-heads' Vahana runs
        print(f"  Vahana's best: {vbest[0]} threads", flush=True)
llama_args = ["--llama-dir", lbin, "--ncmoe", best[0], "--threads", threads, "--ngl", ngl]

# ---------------------------------------------------------------- 3. head-to-heads
out = {"machine": M, "llama.cpp": {"tag": LLAMA_TAG, "n_cpu_moe": best[0], "threads": threads, "ngl": ngl},
       "vahana": {"threads": os.environ.get("VAHANA_THREADS", "default")}}
print("\nbenchmark text (decode after a 512-token prompt)", flush=True)
f1 = os.path.join(RES, "benchmark_text.json")
r = run([sys.executable, os.path.join(EXP, "v15_decode", "head_to_head.py"), "--vahana-model", vdir, "--gguf", gguf,
         *llama_args, "--blocks", a.blocks, "--runs", "3", "--out", f1], log="benchmark_text.log")
def failed(log):
    """A head-to-head that wrote no results: show why, so a failure is never silent."""
    tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-12:]
    print(f"  FAILED (exit {r.returncode}); full log: {os.path.join(RES, log)}", flush=True)
    for line in tail:
        print("    " + line, flush=True)


if not os.path.exists(f1):
    failed("benchmark_text.log")
else:
    out["benchmark_text"] = {k: v for k, v in json.load(open(f1)).items() if k not in ("llama.cpp", "vahana")} | {
        e: {x: y for x, y in json.load(open(f1))[e].items() if x != "runs"} for e in ("llama.cpp", "vahana")}
print("real prompts (decode and time to first token)", flush=True)
f2 = os.path.join(RES, "real_prompts.json")
r = run([sys.executable, os.path.join(EXP, "v15_decode", "real_prompts_h2h.py"), "--vahana-model", vdir, "--gguf", gguf,
         *llama_args, "--blocks", a.blocks, "--out", f2], log="real_prompts.log")
if not os.path.exists(f2):
    failed("real_prompts.log")
else:
    d = json.load(open(f2))
    out["real_prompts"] = {k: v for k, v in d.items() if k not in ("llama.cpp", "vahana")} | {
        e: {x: y for x, y in d[e].items() if x != "runs"} for e in ("llama.cpp", "vahana")}

if a.quality:
    print("quality against the bf16 original", flush=True)
    wiki = os.path.join(a.dir, "wiki.test.raw")
    if not os.path.exists(wiki):
        z = os.path.join(a.dir, "wikitext-2-raw-v1.zip")
        urllib.request.urlretrieve(WIKI, z)
        with zipfile.ZipFile(z) as zf:
            open(wiki, "wb").write(zf.read("wikitext-2-raw/wiki.test.raw"))
    base = os.path.join(a.dir, "bf16_base.dat")
    kld = os.path.join(EXP, "v15_quality", "kld.py")
    if not os.path.exists(base):
        run([sys.executable, kld, "base", bdir, wiki, base, a.chunks], log="quality_base.log")
    rv = run([sys.executable, kld, "score", vdir, base], log="quality_vahana.log")
    rl = run([os.path.join(lbin, "llama-perplexity"), "-m", gguf, "--kl-divergence-base", base, "--kl-divergence",
              "-c", "512", "-ngl", ngl, *(["--n-cpu-moe", best[0]] if ngl != "0" else []), "-t", threads],
             log="quality_llamacpp.log")

    def grab(text, key):
        m = re.search(re.escape(key) + r"\s*:?\s*([-\d.]+)", text)
        return float(m.group(1)) if m else None
    keys = {"ppl": ("Mean PPL(Q)", "Mean PPL(Q)"), "ppl_base": ("Mean PPL(base)", "Mean PPL(base)"),
            "mean_kld": ("Mean KLD", "Mean    KLD"), "kld_99pct": ("99.0% KLD", "99.0%   KLD"),
            "same_top_pct": ("Same top p", "Same top p")}
    out["quality"] = {"vahana": {k: grab(rv.stdout, v[0]) for k, v in keys.items()},
                      "llama.cpp": {k: grab(rl.stdout + rl.stderr, v[1]) for k, v in keys.items()}}

json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)

# ---------------------------------------------------------------- 4. table
print("\n" + "=" * 72)
print(f"{M.get('gpu', 'CPU only')} | {M.get('ram_gb', '?')} GB RAM | {M['os']} | vahana {M['vahana']} | llama.cpp {LLAMA_TAG}")
if "benchmark_text" in out:
    d = out["benchmark_text"]
    print(f"{'benchmark text, decode':24s} llama.cpp {d['llama.cpp']['median']:6.2f} tok/s | "
          f"Vahana {d['vahana']['median']:6.2f} tok/s | {d['ratio_of_medians']:.2f}x")
if "real_prompts" in out:
    d = out["real_prompts"]
    print(f"{'real prompts, decode':24s} llama.cpp {d['llama.cpp']['decode_median']:6.2f} tok/s | "
          f"Vahana {d['vahana']['decode_median']:6.2f} tok/s | {d['decode_ratio_of_medians']:.2f}x")
if "real_prompts" in out:
    d = out["real_prompts"]
    print(f"{'time to first token':24s} llama.cpp {d['llama.cpp']['ttft_median']:6.2f} s     | "
          f"Vahana {d['vahana']['ttft_median']:6.2f} s     | {d['ttft_ratio_of_medians']:.2f}x sooner")
if "quality" in out:
    for e, d in out["quality"].items():
        print(f"quality, {e:15s} " + ", ".join(f"{k} {v}" for k, v in d.items()))
print(f"results: {RES}")
if a.copy_results:
    shutil.copytree(RES, os.path.join(a.copy_results, "vs_llamacpp_results"), dirs_exist_ok=True)
    print(f"copied to {os.path.join(a.copy_results, 'vs_llamacpp_results')}")
