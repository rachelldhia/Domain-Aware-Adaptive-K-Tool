# Domain-Aware Adaptive-k pada WorkBench: kode eksperimen

Implementasi dari proposal (Bab 3). Satu metode saja yang dievaluasi, tanpa baseline.

## Struktur
- `adak/selector.py`: Stage I (gerbang domain ε) dan Stage II (threshold τ, min_k, max_k), persis pseudocode 3.2.5
- `adak/split.py`: split validasi/test per **base_template** (14 / 55 template), terstratifikasi per domain
- `adak/tune.py`: grid ε/τ dari kuantil distribusi skor validasi, metrik offline, Pareto + shortlist, pemilihan akhir (success, lalu token, lalu latency)
- `adak/agent.py`: loop agen (OpenAI-compatible), sandbox WorkBench, evaluator state akhir. Ada `MockAgent` untuk uji pipeline.
- `adak/evaluate.py`: bootstrap per-template (95% CI), H1/H2/H3 (Spearman), kategori kegagalan
- `run_experiment.py`: CLI `prepare -> tune -> validate -> test -> report`

## Menjalankan (real)
```bash
git clone https://github.com/olly-styles/WorkBench && cd WorkBench && git checkout <commit>   # catat & pin commit
pip install -r requirements.txt
# edit config.json: workbench_repo, workbench_commit, encoder.revision (pin hash), agent.model
python run_experiment.py prepare  --out runs/exp1
python run_experiment.py tune     --out runs/exp1
python run_experiment.py validate --out runs/exp1   # memanggil LLM, membekukan frozen_config.json
python run_experiment.py test     --out runs/exp1   # sekali saja; menolak jika sudah pernah / belum freeze
python run_experiment.py report   --out runs/exp1
```
Dry run tanpa jaringan/LLM: tambahkan `--mock` (data sintetis, encoder hashing, agen simulasi). **Output mock bukan hasil penelitian.**

## Yang HARUS diverifikasi terhadap commit WorkBench yang di-pin
Saya tidak bisa mengakses repo saat membuat kode ini, jadi bagian berikut berdasarkan ingatan tentang struktur repo:
1. `adak/data.py` -> `TOOLKITS`: nama `src.tools.toolkits.*_toolkit` dan atribut `.name/.description/.args`.
2. `load_workbench_tasks`: pola file `*queries_and_answers*.csv`, nama kolom (`query`/`answer`/`base_template`/`domains`). Proposal menyebut `task`/`outcome`; kode menerima keduanya.
3. `adak/agent.py` -> `WorkBenchSandbox`: `reset_state()` per modul, `eval("email.send_email.func(...)")`, dan `src.evals.utils.is_correct/has_side_effects` (signature bisa berbeda). Fallback: bandingkan snapshot semua DataFrame modul.
4. System prompt: salin prompt referensi WorkBench ke `config.json -> agent.system_prompt`. Di sini definisi tool dikirim lewat parameter `tools` API function-calling, bukan ditempel di prompt; ini keputusan desain yang perlu Anda tulis di laporan.
5. Hitung ulang Tabel 2-4 proposal (27 tool, 7.396 token) dari commit yang di-pin; `count_tokens` memakai `tiktoken cl100k_base` jika terpasang, jika tidak jatuh ke len/4.

## Keputusan desain yang perlu disepakati kelompok
- `tool_text.split_name=true`: underscore pada nama fungsi diganti spasi sebelum di-embed (membantu MiniLM). Proposal hanya menulis "function name"; dokumentasikan.
- Aturan shortlist: front Pareto atas (write_cov↑, complete_domain↑, def_tokens↓), lalu top-N. Tulis di proposal sebelum menjalankan.
- Metrik offline untuk tuning memakai label domain dan write tool validasi saja (sesuai proposal); test tidak disentuh.
- Task dengan `infra_error` (API/timeout/evaluator) dikeluarkan dari metrik outcome dan dilaporkan jumlahnya.
- Token yang tidak dikembalikan API dicatat `null`, tidak diestimasi.
- Kategori `execution_error_or_read_tool_omission` perlu dicek manual lewat trace (ground truth hanya memuat write action).

## Artefak per run (`runs/<nama>/`)
`split.json`, `grid_offline_validation.csv`, `grid_meta.json`, `shortlist.csv`, `validation_agent_runs.jsonl`, `validation_ranking.csv`, `frozen_config.json`, `test_logs.jsonl`, `test_task_level.csv`, `report.md`
