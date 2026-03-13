import queue
import threading

from homie_core.voice.base_handler import BaseHandler


class EchoHandler(BaseHandler):
    def process(self, item):
        return item.upper() if isinstance(item, str) else item


def test_handler_processes_items():
    in_q, out_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    handler = EchoHandler(in_q, out_q, stop)
    t = threading.Thread(target=handler.run)
    t.start()
    in_q.put("hello")
    assert out_q.get(timeout=2) == "HELLO"
    stop.set()
    in_q.put(b"END")
    t.join(timeout=2)
    assert not t.is_alive()


def test_handler_stops_on_sentinel():
    in_q, out_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    handler = EchoHandler(in_q, out_q, stop)
    t = threading.Thread(target=handler.run)
    t.start()
    in_q.put(b"END")
    t.join(timeout=2)
    assert not t.is_alive()


def test_handler_stops_on_event():
    in_q, out_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    handler = EchoHandler(in_q, out_q, stop)
    t = threading.Thread(target=handler.run)
    t.start()
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()
