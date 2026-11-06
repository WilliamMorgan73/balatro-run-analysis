from balatro_analysis import memory


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv(memory.ENV, "2.5")
    assert memory.run_budget() == 2_500_000_000


def test_budget_share_of_limit(monkeypatch):
    monkeypatch.delenv(memory.ENV, raising=False)
    monkeypatch.setattr(memory, "memory_limit", lambda: 16_000_000_000)
    assert memory.run_budget() == int(16e9 * memory.RUN_SHARE)
    monkeypatch.setattr(memory, "memory_limit", lambda: None)
    assert memory.run_budget() is None


def test_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv(memory.ENV, "lots")
    monkeypatch.setattr(memory, "memory_limit", lambda: 10_000_000_000)
    assert memory.run_budget() == int(10e9 * memory.RUN_SHARE)


def test_slurm_limit(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16384")  # --mem=16G, in MB
    assert memory._slurm_limit() == [16384 << 20]
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    assert memory._slurm_limit() == [(4096 << 20) * 4]


def test_cgroup_limit_files(tmp_path):
    (tmp_path / "limited").write_text("17179869184\n")
    (tmp_path / "unlimited").write_text("max\n")
    assert memory._read_limit(tmp_path / "limited") == 17179869184
    assert memory._read_limit(tmp_path / "unlimited") is None
    assert memory._read_limit(tmp_path / "missing") is None


def test_limit_on_this_machine():
    limit = memory.memory_limit()
    assert limit is None or limit > 0
    rss = memory.process_rss()
    assert rss is None or (limit is None or rss < limit)
