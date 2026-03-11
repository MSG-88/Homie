from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional, Union


class OverlayPopup:
    """Lightweight overlay popup for quick Homie interactions.

    Supports both blocking (on_submit returns str) and streaming
    (on_submit_stream returns Iterator[str]) callbacks.
    """

    def __init__(
        self,
        on_submit: Optional[Callable[[str], str]] = None,
        on_submit_stream: Optional[Callable[[str], Iterator[str]]] = None,
    ):
        self._on_submit = on_submit
        self._on_submit_stream = on_submit_stream
        self._visible = False
        self._root = None
        self._thread: Optional[threading.Thread] = None

    def _handle_submit(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        if self._on_submit:
            return self._on_submit(text.strip())
        return None

    def show(self) -> None:
        if self._visible:
            return
        self._visible = True
        self._thread = threading.Thread(target=self._create_window, daemon=True)
        self._thread.start()

    def hide(self) -> None:
        self._visible = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
            self._root = None

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def _create_window(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            self._visible = False
            return

        self._root = tk.Tk()
        root = self._root
        root.title("Homie")
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        width, height = 600, 200
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 3
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.configure(bg="#1e1e1e")

        input_frame = tk.Frame(root, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        label = tk.Label(input_frame, text="Homie>", fg="#61afef", bg="#1e1e1e",
                         font=("Consolas", 12))
        label.pack(side=tk.LEFT)

        entry = tk.Entry(input_frame, bg="#2d2d2d", fg="white", insertbackground="white",
                         font=("Consolas", 12), relief=tk.FLAT, bd=5)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        entry.focus_set()

        response = tk.Label(root, text="", fg="#abb2bf", bg="#1e1e1e",
                            font=("Consolas", 11), wraplength=560, justify=tk.LEFT,
                            anchor=tk.NW)
        response.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        def on_enter(event=None):
            text = entry.get()
            if not text or not text.strip():
                return
            entry.delete(0, tk.END)
            entry.config(state=tk.DISABLED)

            from homie_app.loading import OverlayLoadingAnimation
            anim = OverlayLoadingAnimation()
            anim.start(root, response)

            def _process():
                try:
                    if self._on_submit_stream:
                        # Streaming: update label as tokens arrive
                        chunks = []
                        first = True
                        for token in self._on_submit_stream(text.strip()):
                            chunks.append(token)
                            display = "".join(chunks)
                            def _update_text(t=display, is_first=first):
                                if is_first:
                                    anim.stop()
                                response.config(text=t)
                            try:
                                root.after(0, _update_text)
                            except Exception:
                                pass
                            first = False
                        # Final update
                        def _done():
                            anim.stop()
                            entry.config(state=tk.NORMAL)
                            entry.focus_set()
                        try:
                            root.after(0, _done)
                        except Exception:
                            pass
                    else:
                        # Blocking fallback
                        result = self._handle_submit(text)
                        def _update():
                            anim.stop()
                            response.config(text=result or "")
                            entry.config(state=tk.NORMAL)
                            entry.focus_set()
                        try:
                            root.after(0, _update)
                        except Exception:
                            pass
                except Exception as e:
                    def _show_error(err=str(e)):
                        anim.stop()
                        response.config(text=f"Error: {err}")
                        entry.config(state=tk.NORMAL)
                        entry.focus_set()
                    try:
                        root.after(0, _show_error)
                    except Exception:
                        pass

            threading.Thread(target=_process, daemon=True).start()

        def on_escape(event=None):
            self.hide()

        entry.bind("<Return>", on_enter)
        root.bind("<Escape>", on_escape)
        root.bind("<FocusOut>", lambda e: None)

        root.mainloop()
        self._visible = False
