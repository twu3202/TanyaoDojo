# scripts

Operation scripts of the value line (Mortal stack) and a few Sichuan runs. Most are tied to the machines they ran on
(paths, proxies); they are kept as the record of how each run was driven. Later experiments keep their scripts next to
their patch under [`patches/`](../patches).

| Group | Scripts |
| --- | --- |
| Data download and repair | `download_all_data.sh`, `download_data_remote.sh`, `dl_server_data.sh`, `dl_all_parallel.sh`, `dl_one_year.sh`, `gzip_2025_2026.sh`, `gzip_fix_local_2025_2026.sh` |
| Offline training pipeline | `queue_train.sh`, `run_track.sh`, `restart_pipeline.sh`, `v18_autostart.sh`, `v18_plateau_watch.sh`, `run_phase1.sh`, `run_phase1b.sh` |
| Online self-play RL | `run_selfplay.sh`, `stop_selfplay.sh`, `switch_v4pool.sh`, `online_dryrun.sh`, `selfplay_smoke.py` |
| Guards and snapshots | `trainer_guard.sh` (restart a vanished trainer), `collapse_guard.py` (stop on two test_play readings below −8), `archiver.sh`, `archiver_steps.sh` (snapshot every 8,000 steps, step-checked), `gpugate.sh`, `gpucheck.py` |
| Evaluation | `eval_watcher.sh`, `gpu_watch_eval.sh`, `srv_evalwatch.sh`, `srv_evalone.sh`, `probe_par.sh`, `run_eval_cprime.sh`, `aggregate_eval.py`, `ako_eval.py` (cross-lineage check against akochan) |
| Analyses and checks | `build_deal_in_table.py`, `verify_v5_obs.py`, `rho_kyoku.py` |
| Mahjax / Sichuan runs | `run_clean.sh`, `resume_clean.sh`, `resume_shaped.sh`, `run_srv3.sh`, `run_p3cap.sh`, `run_p3cap_seed.sh`, `run_sc_cap_eval.sh` |
