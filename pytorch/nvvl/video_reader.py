import logging as log
import torch
from .dataset import ProcessDesc
import collections
import random
import torch.multiprocessing

from . import lib

log_levels = {
    "debug" : lib.LogLevel_Debug,
    "info"  : lib.LogLevel_Info,
    "warn"  : lib.LogLevel_Warn,
    "error" : lib.LogLevel_Error,
    "none"  : lib.LogLevel_None
    }

class VideoReader(object):
    """VideoReader, random read some frames from any position.

        Parameters
        ----------
        filename: collection of strings
            video file name

    """
    def __init__(self, device_id=0, log_level="warn"):
        self.ffi = lib._ffi
        self.tensor_queue = collections.deque()
        self.seq_queue = collections.deque()
        self.processing = None
        self.device_id = device_id
        self.width = None
        self.height = None
        try:
            log_level = log_levels[log_level]
        except KeyError:
            log.info("Invalid log level", log_level, "using warn.")
            log_level = lib.LogLevel_Warn

        self.loader = lib.nvvl_create_video_loader_with_log(device_id, log_level)
        log.info("Success to init VideoReader device_id {}".format(self.device_id))

    def get_frame_count(self, filename):
        return lib.nvvl_video_frame_count_from_file(str.encode(filename))

    def _create_tensor_map(self, batch_size=1):
        tensor_map = {}
        with torch.cuda.device(self.device_id):
            for name, desc in self.processing.items():
                tensor_map[name] = desc.tensor_type(batch_size, *desc.get_dims())
        return tensor_map

    def _set_process_desc(self, width, height, index_map=None):
        self.processing = {"default": ProcessDesc(type='float',
                                                  height=height,
                                                  width=width,
                                                  random_crop=False,
                                                  random_flip=False,
                                                  normalized=False,
                                                  color_space="RGB",
                                                  dimension_order="fhwc",
                                                  index_map=index_map)}

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

    def _start_receive(self, filename, index, length):

        lib.nvvl_read_sequence(self.loader,
                               str.encode(filename),
                               int(index), length)

        seq = lib.nvvl_create_sequence(length)

        # for name, desc in self.processing.items():
        #     desc.count = length

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

    def _finish_reveive(self, synchronous=True):
        if not self.seq_queue:
            raise RuntimeError("Unmatched receive")

        seq = self.seq_queue.popleft()

        if synchronous:
            ret = lib.nvvl_sequence_wait(seq)
        else:
            ret = lib.nvvl_sequence_stream_wait_th(seq)

        lib.nvvl_free_sequence(seq)

        return True if ret == 0 else False

    """Start to read video stream
    Parameter
    ----------
    path: str
        The network address of the video stream.
    """
    def read_stream(self, path):
        if not self.width or self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(path))
            self.height = image_shape.height
            self.width = image_shape.width

        lib.nvvl_read_stream(self.loader, str.encode(path))

    """Get frames data in tensor list
    Parameter
    ----------
    count: int
        The number of frames to get.
    Return
    ----------
    frame_num: int
        The number of the first frame, if encounter the end of stream, return -1
    tensors: list of tensor
        A list of pytorch tensors contain frames data
    """
    def stream_receive(self, count):
        seq = lib.nvvl_create_sequence(count)

        self._set_process_desc(self.width, self.height, index_map=list(range(count)))

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

        lib.nvvl_sequence_stream_wait_th(seq)

        frame_nums = lib.nvvl_get_meta_array(seq, lib.PMT_INT, str.encode("frame_num"))

        frame_nums = self.ffi.cast("int *", frame_nums)
        frame_num = frame_nums[0]

        tensors = []
        for index in range(count):
            tensors.append(tensor_map["default"][0][index].cpu())

        return frame_num, tensors

    def get_samples(self, filename, indexs):
        max_index = max(indexs)
        min_index = min(indexs)

        length = max_index - min_index + 1

        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        index_map = [-1] * length

        for i, index in enumerate(indexs):
            index_map[index - min_index] = i

        self._set_process_desc(self.width, self.height, index_map=index_map)

        # if isinstance(indexs, int):
        #     indexs = [indexs]

        self._start_receive(filename, min_index, length)

        self._finish_reveive()
        t = self.tensor_queue.popleft()
        tensors = t["default"][0].cpu()
        log.info("tensor shape {}".format(t["default"][0].shape))

        return tensors

    def get_samples_old(self, filename, indexs):
        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc(self.width, self.height, index_map=[0])

        if isinstance(indexs, int):
            indexs = [indexs]

        for index in indexs:
            self._start_receive(filename, index, 1)

        tensors = []
        ret = True
        for index in indexs:
            r = self._finish_reveive()
            if not r:
                ret = False
            t = self.tensor_queue.popleft()
            tensors.append(t["default"][0].cpu())

        return tensors if ret else None

    def get_samples_old_sync(self, filename, indexs):
        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc(self.width, self.height, index_map=[0])

        if isinstance(indexs, int):
            indexs = [indexs]

        tensors = []
        for index in indexs:
            self._start_receive(filename, index, 1)

            self._finish_reveive(synchronous=True)
            t = self.tensor_queue.popleft()
            tensors.append(t["default"][0].cpu())

        return tensors

    def get_frames(self, filename, index, length):
        """get n {length} frams from position {index} in video

        Parameters
        ----------
        index : list of int or int
            index list or index of the needed frame
        length:
            frames length
        """

        lib.nvvl_read_sequence(self.loader,
                               str.encode(filename),
                               index, length)

        log.info("Start to read sequence from index {}, length {}".format(index, length))

        seq = lib.nvvl_create_sequence(length)

        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc(self.width, self.height)

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

    def finish(self):
        lib.nvvl_finish_video_loader(self.loader)

    def destroy(self):
        lib.nvvl_destroy_video_loader(self.loader)


