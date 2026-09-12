"""One standard image and one Python-random prefix; no second view or suffix output."""
import os
import random
from PIL import Image
try:
    from .said_cvssl_data import Share4VCvsslDataset, image_id_from_path
except ImportError:
    from said_cvssl_data import Share4VCvsslDataset, image_id_from_path


def sample_prefix(caption):
    sentences = caption.replace('\n', ' ').split('. ')
    k = random.randint(1, len(sentences))
    return '. '.join(sentences[:k]), k


class PrefixDataset(Share4VCvsslDataset):
    def __getitem__(self, index):
        item = self.json_data[index]
        prefix, k = sample_prefix(item['conversations'][1]['value'])
        with Image.open(os.path.join(self.image_root, item['image'])) as image:
            image_a = self.view_a(image.convert('RGB'))
        return {'image_a': image_a, 'caption_said': prefix, 'prefix_k': k,
                'image_id': image_id_from_path(item['image']), 'sample_id': index + self.total_len}
