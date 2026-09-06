import unittest

from cache import Cache


class CacheTests(unittest.TestCase):
    def test_round_trip(self):
        cache = Cache()
        cache.put("a", "hello", ttl=5, now=10)
        self.assertEqual(cache.get("a", now=14), "hello")
        self.assertIsNone(cache.get("a", now=16))


if __name__ == "__main__":
    unittest.main()
