"""A small cache with caller-supplied time, for deterministic tests."""


class Cache:
    def __init__(self):
        self._items = {}

    def put(self, key, value, ttl, now):
        self._items[key] = (value, now + ttl)

    def get(self, key, now, default=None):
        item = self._items.get(key)
        if item is None:
            return default
        value, expires = item
        if now > expires:
            del self._items[key]
            return default
        return value
