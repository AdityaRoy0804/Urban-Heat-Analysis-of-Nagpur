import unittest

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment; install requirements.txt to run")
class TestAPIHealth(unittest.TestCase):
    def setUp(self):
        from nagpur_uhi.api.main import app

        self.client = TestClient(app)

    def test_health_endpoint_responds(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_list_zones_endpoint(self):
        response = self.client.get("/zones")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_zones", data)
        self.assertIn("zones", data)

    def test_unknown_zone_returns_404(self):
        response = self.client.get("/zones/does_not_exist/history")
        self.assertIn(response.status_code, (404, 503))  # 503 if no dataset loaded at all

    def test_root_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))


if __name__ == "__main__":
    unittest.main()
