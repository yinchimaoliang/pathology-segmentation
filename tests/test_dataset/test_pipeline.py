from pathseg.datasets.pipelines import Loading


def test_loading():
    loading = Loading()

    results = dict(
        img_path='./tests/data/images/1.png',
        ann_path='./tests/data/annotations/1.png')

    results = loading(results)

    image = results['image']
    annotation = results['annotation']

    assert image.shape == (5736, 3538, 3)
    assert annotation.shape == (5736, 3538)
