from pathlib import Path

from scripts.simulate_requests import iter_request_images


def test_iter_request_images_generates_labeled_synthetic_batch(tmp_path: Path) -> None:
    requests = list(iter_request_images(tmp_path))

    assert [(name, label) for name, label, _ in requests] == [
        ("simulated-cat.jpg", "cat"),
        ("simulated-dog.jpg", "dog"),
    ]
    assert all(image_bytes.startswith(b"\xff\xd8") for _, _, image_bytes in requests)
