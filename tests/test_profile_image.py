# [S:TEST v1] profile image exists and contains title pass
from pathlib import Path


def test_profile_image_exists_and_has_text():
    img_path = Path('docs/images/rft_behavioral_approach.svg')
    assert img_path.is_file(), 'profile image missing'
    content = img_path.read_text(encoding='utf-8')
    assert '<svg' in content and 'Behavioral RFT Approach' in content
