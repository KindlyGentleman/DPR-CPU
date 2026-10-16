import pytest
from cpu_optimization.benchmarks.threading import WorkerProxy


def test_worker_uses_prepared_input_identity_not_payload_timing():
    class Tensor:
        def data_ptr(self):
            return 123

    class Connection:
        def send(self, value):
            self.value = value

        def poll(self, timeout):
            return True

        def recv(self):
            return {"seconds": 0.25, "prediction": [[1.0]], "memory": {"rss": 5}}

    conn = Connection()
    worker = WorkerProxy(conn, {123: 2})
    out = worker(Tensor(), Tensor(), false_scale=False)
    assert conn.value == {"op": "predict", "index": 2}
    assert worker.last["seconds"] == 0.25
    assert out.shape == (1, 1, 1, 1)
    with pytest.raises(ValueError):
        worker(Tensor(), Tensor(), false_scale=True)
