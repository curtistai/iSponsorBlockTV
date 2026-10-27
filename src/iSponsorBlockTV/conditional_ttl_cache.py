import asyncio
import time
from functools import wraps

from cache.lru import LRU

# MIT License

# Copyright (c) 2020 Rajat Singh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified code from https://github.com/iamsinghrajat/async-cache


class AsyncConditionalTTL:
    class _TTL(LRU):
        def __init__(self, time_to_live, maxsize):
            super().__init__(maxsize=maxsize)

            self.time_to_live = time_to_live

            self.maxsize = maxsize

        def __contains__(self, key):
            if key not in self.keys():
                return False
            key_expiration = super().__getitem__(key)[1]
            if key_expiration and key_expiration < time.monotonic():
                del self[key]
                return False
            return True

        def __getitem__(self, key):
            value = super().__getitem__(key)[0]
            return value

        def __setitem__(self, key, value):
            value, ignore_ttl = value  # unpack tuple
            ttl_value = (
                (time.monotonic() + self.time_to_live)
                if (self.time_to_live and not ignore_ttl)
                else None
            )  # ignore ttl if ignore_ttl is True
            super().__setitem__(key, (value, ttl_value))

    def __init__(self, time_to_live=60, maxsize=1024, skip_args: int = 0):
        """

        :param time_to_live: Use time_to_live as None for non expiring cache
        :param maxsize: Use maxsize as None for unlimited size cache
        :param skip_args: Use `1` to skip first arg of func in determining cache key
        """
        self.ttl = self._TTL(time_to_live=time_to_live, maxsize=maxsize)
        self.skip_args = skip_args

    def __call__(self, func):
        pending = {}

        async def fetch(key, args, kwargs):
            result = await func(*args, **kwargs)
            self.ttl[key] = result
            return result[0]

        def finished(key, task):
            pending.pop(key, None)
            # Retrieve failures even when every caller has been cancelled.
            if not task.cancelled():
                task.exception()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = (args[self.skip_args :], tuple(sorted(kwargs.items())))
            if key in self.ttl:
                return self.ttl[key]
            task = pending.get(key)
            if task is None:
                task = asyncio.create_task(fetch(key, args, kwargs))
                pending[key] = task
                task.add_done_callback(lambda done: finished(key, done))
            # Playback updates may cancel one caller without cancelling shared prefetches.
            return await asyncio.shield(task)

        return wrapper
