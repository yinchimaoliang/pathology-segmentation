import cv2 as cv
import numpy as np

from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Flip(object):

    def __init__(self,
                 prob=.5,
                 flip_ratio_horizontal=.5,
                 flip_ratio_vertical=.5):
        self.prob = prob
        self.flip_ratio_horizontal = flip_ratio_horizontal
        self.flip_ratio_vertical = flip_ratio_vertical

    def __call__(self, results):
        img = results['image']
        ann = results['annotation']

        results['flip_horizontal'] = False
        results['flip_ratio_vertical'] = False
        if np.random.random() < self.prob:
            if np.random.random() < self.flip_ratio_horizontal:
                img = np.ascontiguousarray(img[:, ::-1, ...])
                ann = np.ascontiguousarray(ann[:, ::-1, ...])
                results['flip_horizontal'] = True

            if np.random.random() < self.flip_ratio_vertical:
                img = np.ascontiguousarray(img[::-1, ...])
                ann = np.ascontiguousarray(ann[::-1, ...])
                results['flip_ratio_vertical'] = True

            results['image'] = img
            results['annotation'] = ann

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(prob={})'.format(self.prob)
        repr_str += '(flip_ratio_horizontal={})'.format(
            self.flip_ratio_horizontal)
        repr_str += '(flip_ratio_vertical={})'.format(self.flip_ratio_vertical)
        return repr_str


@PIPELINES.register_module()
class ShiftScaleRotate(object):

    def __init__(self,
                 prob=.5,
                 rotate_range=[-15, 15],
                 scale_range=[0.9, 1.1]):
        self.prob = prob
        self.rotate_range = rotate_range
        self.scale_range = scale_range

    def _maybe_process_in_chunks(self, process_fn, **kwargs):
        """
        Wrap OpenCV function to enable processing images
            with more than 4 channels.

        Limitations:
            This wrapper requires image to be the first argument
                and rest must be sent via named arguments.

        Args:
            process_fn: Transform function (e.g cv2.resize).
            kwargs: Additional parameters.

        Returns:
            numpy.ndarray: Transformed image.

        """

        def __process_fn(img):
            num_channels = img.shape[2] if len(img.shape) == 3 else 1
            if num_channels > 4:
                chunks = []
                for index in range(0, num_channels, 4):
                    chunk = img[:, :, index:index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
                img = np.dstack(chunks)
            else:
                img = process_fn(img, **kwargs)
            return img

        return __process_fn

    def _shift_scale_rotate(self,
                            img,
                            angle,
                            scale,
                            dx=0,
                            dy=0,
                            interpolation=cv.INTER_LINEAR,
                            border_mode=cv.BORDER_REFLECT_101,
                            value=None):
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height

        warp_affine_fn = self._maybe_process_in_chunks(
            cv.warpAffine,
            M=matrix,
            dsize=(width, height),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=value)
        return warp_affine_fn(img)

    def __call__(self, results):
        results['shift_scale_rotate'] = False
        if np.random.random() < self.prob:
            img = results['image']
            ann = results['annotation']

            angle = np.random.uniform(self.rotate_range[0],
                                      self.rotate_range[1])
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

            img = self._shift_scale_rotate(img, angle, scale)
            ann = self._shift_scale_rotate(
                ann, angle, scale, interpolation=cv.INTER_NEAREST)

            results['image'] = img
            results['annotation'] = ann
            results['shift_scale_rotate'] = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(prob={})'.format(self.prob)
        repr_str += '(rotate_range={})'.format(self.rotate_range)
        repr_str += '(scale_range={})'.format(self.scale_range)
        return repr_str


@PIPELINES.register_module()
class RandomRotate90(object):

    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, results):
        results['random_rotate_90'] = False
        if np.random.random() < self.prob:
            img = results['image']
            ann = results['annotation']

            factor = np.random.randint(0, 3)

            img = np.ascontiguousarray(np.rot90(img, factor))
            ann = np.ascontiguousarray(np.rot90(ann, factor))

            results['image'] = img
            results['annotation'] = ann

            results['random_rotate_90'] = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(prob={})'.format(self.prob)
        return repr_str
