"""Converting while downloading: a checkpoint's shards are fetched in the order a converter reads them, a little
ahead of it, and each shard is deleted once the converter has read past it. Converting MiniMax-H3-Image then needs
room for its output and three shards (~17 GB) instead of the whole checkpoint as well (~125 GB).

The converters read monotonically through a sharded checkpoint (MiniMax-H3-Image's transformer and text encoder are
sharded in layer order), with at most one shard of look-back where a layer straddles two shards. StreamedSharded
keeps that one and deletes the rest; a read from a deleted shard is an error, not a silent re-download.
"""

from __future__ import annotations

import os
import threading

from .st import SafeTensors, Sharded


class ShardStream:
    """Downloads files [(path, size, sha256, url)] in the given order into dest, at most `ahead` files beyond the
    furthest one asked for, in a background thread."""

    def __init__(self, files, dest: str, ahead: int = 1, log=print):
        self.files = list(files)
        self.order = {f[0]: i for i, f in enumerate(self.files)}
        self.dest, self.ahead, self.log = dest, ahead, log
        self.done: dict[str, tuple[bool, str]] = {}
        self.released: set[str] = set()
        self.furthest = 0  # the furthest file asked for: downloads run up to furthest + ahead
        self.stopped = False
        self.peak_files = 0  # the most shard files present at once (tests check the window)
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _present(self) -> int:
        return sum(1 for p in self.done if p not in self.released and os.path.exists(os.path.join(self.dest, p)))

    def _run(self):
        from .download import _fetch
        for i, item in enumerate(self.files):
            with self.cv:
                while i > self.furthest + self.ahead and not self.stopped:
                    self.cv.wait()
                if self.stopped:
                    return
            try:
                path, ok, msg = _fetch(item, self.dest)
            except Exception as e:  # recorded, so that wait() raises instead of waiting for ever
                path, ok, msg = item[0], False, f"{type(e).__name__}: {e}"
            with self.cv:
                self.done[path] = (ok, msg)
                self.peak_files = max(self.peak_files, self._present())
                self.cv.notify_all()
            self.log(f"  {'ok ' if ok else 'ERR'} {path} ({msg})")
            if not ok:
                return

    def wait(self, path: str) -> str:
        """The local file for path, once downloaded (raises if its download failed)."""
        i = self.order[path]
        with self.cv:
            self.furthest = max(self.furthest, i)
            self.cv.notify_all()
            while path not in self.done:
                failed = [p for p, (ok, _) in self.done.items() if not ok]
                if failed:
                    raise RuntimeError(f"the download of {failed[0]} did not complete; run the command again to "
                                       "resume")
                if not self.thread.is_alive():
                    raise RuntimeError(f"the downloader stopped before {path}; run the command again to resume")
                self.cv.wait(timeout=5)
            ok, msg = self.done[path]
        if not ok:
            raise RuntimeError(f"{path}: {msg}; run the command again to resume")
        return os.path.join(self.dest, path)

    def release(self, path: str):
        with self.cv:
            self.released.add(path)
            f = os.path.join(self.dest, path)
            if os.path.exists(f):
                os.remove(f)
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()
        self.thread.join(timeout=10)


class StreamedSharded(Sharded):
    """A sharded checkpoint whose shards come from a ShardStream and go once read past (one shard of look-back).
    `prefix` is the directory's path in the stream (e.g. "transformer/")."""

    def __init__(self, directory: str, stream: ShardStream, prefix: str):
        super().__init__(directory)  # the index only: no shard is opened yet
        self.stream, self.prefix = stream, prefix

    def _file(self, name: str) -> SafeTensors:
        fn = self.where[name]
        if fn not in self.files:
            key = self.prefix + fn
            if key in self.stream.released:
                raise RuntimeError(f"{fn} was read past and deleted before {name} was read (the converter's read "
                                   "order is not the stream's)")
            self.files[fn] = SafeTensors(self.stream.wait(key))
            here = self.stream.order[key]
            for other in list(self.files):
                if self.stream.order[self.prefix + other] < here - 1:
                    del self.files[other]  # drop the mapping, then the file
                    self.stream.release(self.prefix + other)
        return self.files[fn]

    def close(self):
        """Drops the open shards and deletes every shard the stream fetched for this directory, opened or not
        (one fetched ahead that was never needed, for instance)."""
        self.files.clear()
        for path in list(self.stream.done):
            if path.startswith(self.prefix):
                self.stream.release(path)
