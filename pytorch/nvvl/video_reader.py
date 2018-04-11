import logging as log
import torch
from .dataset import ProcessDesc
import collections
import random

from . import lib

class VideoReader(object):
    """VideoReader, random read some frames from any position.

        Parameters
        ----------
        filename: collection of strings
            video file name

    """
    def __init__(self, filename):
        self.ffi = lib._ffi
        self.filename = filename
        self.image_shape = lib.nvvl_video_size_from_file(self.filename.encode('ascii'))
        self.height = self.image_shape.height
        self.width = self.image_shape.width
        self.tensor_queue = collections.deque()
        self.seq_queue = collections.deque()
        self.processing = {"default": ProcessDesc(type='float',
                                                 height=self.height,
                                                 width=self.width,
                                                 random_crop=False,
                                                 random_flip=False,
                                                 normalized=False,
                                                 color_space="RGB",
                                                 dimension_order="fhwc",
                                                 index_map=None)}
        self.loader = lib.nvvl_create_video_loader(0)
        log.info("Success to init VideoReader for {} {}x{}".format(self.filename, self.width, self.height))

    def _create_tensor_map(self, batch_size=1):
        tensor_map = {}
        for name, desc in self.processing.items():
            tensor_map[name] = desc.tensor_type(batch_size, *desc.get_dims())
        return tensor_map

    def _get_layer_desc(self, desc):
        d = desc.desc()

        if (desc.random_crop and (self.width > desc.width)):
            d.crop_x = random.randint(0, self.width - desc.width)
        else:
            d.crop_x = 0

        if (desc.random_crop and (self.height > desc.height)):
            d.crop_y = random.randint(0, self.height - desc.height)
        else:
            d.crop_y = 0

        if (desc.random_flip):
            d.horiz_flip = random.random() < 0.5
        else:
            d.horiz_flip = False

        return d

    def _start_receive(self, index, length):

        lib.nvvl_read_sequence(self.loader,
                               str.encode(self.filename),
                               index, length)

        seq = lib.nvvl_create_sequence(length)

        for name, desc in self.processing.items():
            desc.count = length

        tensor_map = self._create_tensor_map()

        for name, desc in self.processing.items():
            tensor = tensor_map[name]
            layer = self.ffi.new("struct NVVL_PicLayer*")
            if desc.tensor_type == torch.cuda.FloatTensor:
                layer.type = lib.PDT_FLOAT
            elif desc.tensor_type == torch.cuda.HalfTensor:
                layer.type = lib.PDT_HALF
            elif desc.tensor_type == torch.cuda.ByteTensor:
                layer.type = lib.PDT_BYTE

            # log.info("tensor {}".format(tensor))

            strides = tensor[0].stride()
            try:
                desc.stride.x = strides[desc.dimension_order.index('w')]
                desc.stride.y = strides[desc.dimension_order.index('h')]
                desc.stride.n = strides[desc.dimension_order.index('f')]
                desc.stride.c = strides[desc.dimension_order.index('c')]
            except ValueError:
                raise ValueError("Invalid dimension order")
            layer.desc = self._get_layer_desc(desc)[0]
            if desc.index_map_length > 0:
                layer.index_map = desc.index_map
                layer.index_map_length = desc.index_map_length
            else:
                layer.index_map = self.ffi.NULL
            layer.data = self.ffi.cast("void*", tensor[0].data_ptr())
            lib.nvvl_set_layer(seq, layer, str.encode(name))

        lib.nvvl_receive_frames(self.loader, seq)
        self.tensor_queue.append(tensor_map)
        self.seq_queue.append(seq)

    def _finish_reveive(self, synchronous=False):
        if not self.seq_queue:
            raise RuntimeError("Unmatched receive")

        seq = self.seq_queue.popleft()

        if synchronous:
            lib.nvvl_sequence_wait(seq)
        else:
            lib.nvvl_sequence_stream_wait_th(seq)
        lib.nvvl_free_sequence(seq)

    def get_samples(self, indexs, length):

        if isinstance(indexs, int):
            indexs = [indexs]

        for index in indexs:
            self._start_receive(index, length)

        tensors = []
        for index in indexs:
            self._finish_reveive()
            t = self.tensor_queue.popleft()
            tensors.append(t["default"][0].cpu())

        return tensors

    def get_frames(self, index, length):
        """get n {length} frams from position {index} in video

        Parameters
        ----------
        index : list of int or int
            index list or index of the needed frame
        length:
            frames length
        """

        lib.nvvl_read_sequence(self.loader,
                               str.encode(self.filename),
                               index, length)

        log.info("Start to read sequence from index {}, length {}".format(index, length))

        seq = lib.nvvl_create_sequence(length)

        for name, desc in self.processing.items():
            desc.count = length

        tensor_map = self._create_tensor_map()

        for name, desc in self.processing.items():
            tensor = tensor_map[name]
            layer = self.ffi.new("struct NVVL_PicLayer*")
            if desc.tensor_type == torch.cuda.FloatTensor:
                layer.type = lib.PDT_FLOAT
            elif desc.tensor_type == torch.cuda.HalfTensor:
                layer.type = lib.PDT_HALF
            elif desc.tensor_type == torch.cuda.ByteTensor:
                layer.type = lib.PDT_BYTE

            # log.info("tensor {}".format(tensor))

            strides = tensor[0].stride()
            try:
                desc.stride.x = strides[desc.dimension_order.index('w')]
                desc.stride.y = strides[desc.dimension_order.index('h')]
                desc.stride.n = strides[desc.dimension_order.index('f')]
                desc.stride.c = strides[desc.dimension_order.index('c')]
            except ValueError:
                raise ValueError("Invalid dimension order")
            layer.desc = self._get_layer_desc(desc)[0]
            if desc.index_map_length > 0:
                layer.index_map = desc.index_map
                layer.index_map_length = desc.index_map_length
            else:
                layer.index_map = self.ffi.NULL
            layer.data = self.ffi.cast("void*", tensor[0].data_ptr())
            lib.nvvl_set_layer(seq, layer, str.encode(name))

        lib.nvvl_receive_frames_sync(self.loader, seq)

        if len(tensor_map) == 1 and "default" in tensor_map:
            return tensor_map["default"][0].cpu()
        return {name: tensor[0].cpu() for name, tensor in tensor_map.items()}

    def destory(self):
        lib.nvvl_destroy_video_loader(self.loader)


