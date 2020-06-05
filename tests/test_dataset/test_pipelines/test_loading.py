from pathseg.datasets.pipelines import Loading


def test_loading():
    loading = Loading(shape=(255, 255))

    results = dict(
        img_path='./tests/data/images/test.png',
        ann_path='./tests/data/annotations/test.png')

    results = loading(results)

    image = results['image']
    annotation = results['annotation']

    assert image.shape == (255, 255, 3)
    assert annotation.shape == (255, 255)
