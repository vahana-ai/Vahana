"""Run vs_llamacpp.py on Modal (https://modal.com): one T4, 4 CPUs, 32 GB RAM, like Kaggle's T4 machines.

  pip install modal && modal setup           # once: signs you in to your Modal account in the browser
  modal run experiments/reproduce/vs_modal.py --model mixtral-8x7b

The container installs Vahana from GitHub, downloads both models into its own disk, runs the comparison and prints
the table. results/ (JSON and logs) is returned and written to ./modal_results/ on your machine. A T4 run of about
3 hours costs a few dollars of Modal's monthly free credits.
"""
import io
import os
import tarfile

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgomp1", "libcurl4")
    .pip_install("vahana[gpu] @ git+https://github.com/vahana-ai/Vahana", "huggingface_hub")
    .run_commands("git clone -q https://github.com/vahana-ai/Vahana /opt/Vahana")
)
app = modal.App("vahana-vs-llamacpp", image=image)


@app.function(gpu="T4", cpu=4, memory=32768, timeout=6 * 3600)
def compare(model: str = "qwen3-30b") -> bytes:
    import subprocess
    subprocess.run(["git", "-C", "/opt/Vahana", "pull", "-q"], check=False)       # the newest script
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; nproc; df -h /tmp", shell=True)
    subprocess.run("vahana selftest --device cuda 2>&1 | tail -1", shell=True)
    subprocess.run(["python", "/opt/Vahana/experiments/reproduce/vs_llamacpp.py", "--model", model,
                    "--dir", "/tmp/vs", "--yes"])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if os.path.isdir("/tmp/vs/results"):
            tf.add("/tmp/vs/results", arcname="results")
    return buf.getvalue()


@app.local_entrypoint()
def main(model: str = "qwen3-30b"):
    data = compare.remote(model)
    os.makedirs("modal_results", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall("modal_results", filter="data")
    print("results written to ./modal_results/results")
