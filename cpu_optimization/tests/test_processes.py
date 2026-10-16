from cpu_optimization.scripts import campaign


def test_timeout_cleanup_is_limited_to_owned_tree_and_children_first(monkeypatch):
    calls=[]
    class Process:
        def __init__(self,pid): self.pid=pid
        def children(self,recursive): return [Process(43),Process(44)]
        def terminate(self): calls.append(self.pid)
    monkeypatch.setattr(campaign.psutil,'Process',Process)
    monkeypatch.setattr(campaign.psutil,'wait_procs',lambda processes,timeout:(processes,[]))
    campaign.terminate_tree(42)
    assert calls==[44,43,42]
