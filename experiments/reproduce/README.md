# Reproduce Vahana vs llama.cpp

One command runs the whole comparison on your machine (`vs_llamacpp.py`):

```bash
git clone https://github.com/vahana-ai/Vahana
pip install "vahana[gpu] @ git+https://github.com/vahana-ai/Vahana"     # or "vahana @ git+..." without a GPU
python Vahana/experiments/reproduce/vs_llamacpp.py                       # --quality adds the bf16 comparison
```

The script works in four steps:
1. **Download, after asking.** It fetches Vahana's model (RedHatAI's 4-bit Qwen3-30B-A3B, about 16.5 GB) and llama.cpp's model (Qwen's Q4_K_M GGUF, about 18.6 GB). It also fetches the llama.cpp b12494 release built for your OS and GPU.
2. **Tune llama.cpp.** It sweeps `--n-cpu-moe` with `llama-bench` and keeps llama.cpp's fastest setting on this machine. Vahana runs with its defaults.
3. **Run both engines on the same prompts.** They alternate, both warm:
   - decode speed on a benchmark text;
   - decode speed and time to first token on five real prompts (code, chat, math, story, Hindi).

   With `--quality`, both 4-bit models are also scored against the bf16 original on the same WikiText-2 tokens. This downloads the bf16 model, about 61 GB more.
4. **Report.** It prints a table and writes JSON and logs to `~/vahana_vs_llamacpp/results`, along with a description of the machine.

`--model mixtral-8x7b` runs the same comparison on Mixtral-8x7B-Instruct: Vahana's model is `nm-testing/Mixtral-8x7B-Instruct-v0.1-W4A16-quantized` (about 24.5 GB) and llama.cpp's is `mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF`, Q4_K_M (about 28.4 GB).

The protocols are described in [experiments/v15_decode](../v15_decode/README.md) and [experiments/v15_quality](../v15_quality/README.md).

## On a free Kaggle GPU (T4, 16 GB)

1. Create a new notebook.
   - Settings → Accelerator: **GPU T4 x2** (the script uses one).
   - Settings → Internet: **on**. Kaggle needs a phone-verified account for this.
2. Run these cells:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; nproc; df -h /tmp /kaggle/working
!pip install -q "vahana[gpu] @ git+https://github.com/vahana-ai/Vahana"
!git clone -q https://github.com/vahana-ai/Vahana /kaggle/working/Vahana
!CUDA_VISIBLE_DEVICES=0 vahana selftest --device cuda
```

```python
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/Vahana/experiments/reproduce/vs_llamacpp.py --dir /tmp/vs --yes --copy-results /kaggle/working 2>&1 | tail -40
```

The second cell downloads about 35 GB and runs for about an hour. The models go to `/tmp`, because `/kaggle/working` holds only 20 GB. The results are copied to `/kaggle/working/vs_llamacpp_results`.

## Results so far

**Google Colab, free T4 (15 GB), 12.7 GB RAM, 1 CPU core (2 threads)**, run with this script (`results_colab_t4.json`):
- **llama.cpp's tuning.** Its best was `--n-cpu-moe 12 -t 1`. The sweep ran 0.98 tok/s with all 48 layers' experts on the CPU, 5.15 with 12, and 8 did not fit the GPU. Two threads were slower (4.45).
- **Vahana.** Its defaults, plus the adaptive GPU expert cache.

| | llama.cpp | Vahana | |
|---|---|---|---|
| benchmark text, decode (6 runs each, ranges 8.09–9.37 vs 16.82–17.23) | 8.70 tok/s | **17.05 tok/s** | **1.96×** |
| five real prompts, decode (median of 10 each) | 5.37 tok/s | **8.85 tok/s** | **1.65×** |
| time to first token, real prompts (median) | 12.76 s | **3.71 s** | **3.44× sooner** |

- On the real prompts Vahana was faster on 9 of the 10 prompt runs. The code prompt in block 0 was level: 4.72 vs 4.79 tok/s.
- In the same run, the T4 passed `vahana selftest --device cuda`: all 18 families gave the reference machine's bits, although the T4 has no native bf16.

**Laptop, RTX 3050 Ti (4 GB), 16 GB RAM**:
- **Decode:** 1.86× on the benchmark text, and 1.57× on the real prompts.
- **Time to first token:** 2.8× sooner.
- See [experiments/v15_decode](../v15_decode/README.md), section 9.

**Kaggle, CPU only: 2 cores (4 threads), 31 GB RAM, no GPU** (Kaggle assigns an Intel Xeon 2.2 GHz or an AMD EPYC 7B12 per session; this run's model was not recorded) (`results_kaggle_cpu.json`). **llama.cpp wins here, clearly.**
- **Tuning.** Both engines were tuned on this machine. llama.cpp ran 4.97 tok/s with 2 threads and 7.92 with 4. Vahana ran 1.20 with 2 threads and 1.38 with 4. Both used 4.

| | llama.cpp | Vahana | |
|---|---|---|---|
| benchmark text, decode | **6.35 tok/s** | 1.35 tok/s | 0.21× |
| five real prompts, decode | **5.43 tok/s** | 1.34 tok/s | 0.25× |
| time to first token, real prompts | **11.6 s** | 13.9 s | llama.cpp 1.2× sooner |

- **After the CPU work in 1.4** (4-bit attention linears on the CPU, a one-call RMSNorm; same bits), a second Kaggle run, again both engines tuned (that session got an AMD EPYC 7B12, 2 cores / 4 threads, so compare each run's own ratio): decode 2.21 vs 7.91 tok/s on the benchmark text (**0.28×**, from 0.21×), 2.00 vs 6.88 on real prompts (**0.29×**, from 0.25×), first token 10.3 vs 8.8 s. llama.cpp still wins clearly without a GPU. Two later changes (kernel calls skip TwinMode's dispatch; one thread per physical core) are not in this run.
- **Why.** Vahana's wins come from splitting work between a GPU and the CPU. With no GPU, every layer runs on the CPU in Vahana's fixed summation order, so the bits match every other machine. A first profile on the laptop with its GPU hidden (4 threads, about 460 ms per decode step):
  - about 115 ms is PyTorch overhead: about 5,400 small operations per token (slices, adds, copies), the glue around the exact kernels; on a GPU, CUDA graphs hide it;
  - 87 ms is the expert kernel, with the weights already in RAM;
  - the remaining ~250 ms is Vahana's exact dense kernels plus the laptop's paging (this profile does not separate them).

  So the CPU-only path has not been tuned the way the GPU path has. It is a known gap, not a limit of exact arithmetic.
- **First run.** Vahana did not start in this container at first: Kaggle forbids the `set_mempolicy` call it uses to switch off NUMA balancing, as Docker's default profile does. That is fixed in 1.4.0; the call is now a hint, so a refusal is logged and the run continues.
- **Exactness held.** `vahana selftest --device cpu` gave the reference machine's bits on all 18 families here.

**Mixtral-8x7B on a free Colab T4 (12.7 GB RAM): not measurable there.** Both 4-bit files are about twice the RAM (Vahana's 24.5 GB, llama.cpp's 28.4 GB), so both engines read weights from disk on nearly every token. llama.cpp's best setting reached only 0.17 tok/s (`--n-cpu-moe 16 -t 2`; fewer CPU layers did not fit the GPU). The two head-to-heads did not finish within the scripts' time limits, and Colab recycled the runtime before the logs could be read, so there is no Vahana number. A fair Mixtral comparison needs about 32 GB of RAM (Kaggle's T4 machines have 31 GB); that run is pending.

**Quality against the bf16 original** (the same on any machine): Vahana's 4-bit has mean KL divergence 0.065 against llama.cpp Q4_K_M's 0.093. See [experiments/v15_quality](../v15_quality/README.md).
