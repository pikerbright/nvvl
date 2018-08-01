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

class PreProcess(object):
    def __init__(self, crop_width=0, crop_height=0, scale_size=0,
                 random_crop=False, random_flip=False, normalized=False):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.scale_size = scale_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.normalized = normalized
         

class VideoReader(object):
    """VideoReader, random read some frames from any position.

        Parameters
        ----------
        filename: collection of strings
            video file name

    """
    def __init__(self, device_id=0, preprocess=None, log_level="warn"):
        self.ffi = lib._ffi
        self.tensor_queue = collections.deque()
        self.seq_queue = collections.deque()
        self.processing = None
        self.device_id = device_id
        self.width = None
        self.height = None
        if preprocess is None:
            self.preprocess = PreProcess()
        else:
            self.preprocess = preprocess

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

    def _set_process_desc(self, index_map=None):
        width = self.preprocess.crop_width if self.preprocess.crop_width else self.width
        height = self.preprocess.crop_height if self.preprocess.crop_height else self.height
        if isinstance(self.preprocess.scale_size, int):
            if self.width < self.height:
                scale_width = self.preprocess.scale_size
                scale_height = int(self.preprocess.scale_size * self.height / self.width)
            else:
                scale_height = self.preprocess.scale_size
                scale_width = int(self.preprocess.scale_size * self.width / self.height)
        else:
            scale_width, scale_height = self.preprocess.scale_size

        self.processing = {"default": ProcessDesc(type='float',
                                                  height=height,
                                                  width=width,
                                                  scale_width=scale_width,
                                                  scale_height=scale_height,
                                                  random_crop=self.preprocess.random_crop,
                                                  random_flip=self.preprocess.random_flip,
                                                  normalized=self.preprocess.normalized,
                                                  color_space="RGB",
                                                  dimension_order="fhwc",
                                                  index_map=index_map)}

    def _get_layer_desc(self, desc):
        d = desc.desc()

        width = self.width
        height = self.height
        if desc.scale_width:
            d.scale_width = desc.scale_width
            width = desc.scale_width

        if desc.scale_height:
            d.scale_height = desc.scale_height
            height = desc.scale_height

        if (desc.random_crop and (width > desc.width)):
            d.crop_x = random.randint(0, width - desc.width)
        elif width > desc.width:
            d.crop_x = int((width - desc.width + 1) / 2)
        else:
            d.crop_x = 0

        if (desc.random_crop and (height > desc.height)):
            d.crop_y = random.randint(0, height - desc.height)
        elif height - desc.height:
            d.crop_y = int((height - desc.height + 1) / 2)
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

        seq = lib.nvvl_create_sequence_device(length, self.device_id)

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

    def _finish_reveive(self, synchronous=False):
        if not self.seq_queue:
            raise RuntimeError("Unmatched receive")

        seq = self.seq_queue.popleft()

        if synchronous:
            lib.nvvl_sequence_wait(seq)
        else:
            lib.nvvl_sequence_stream_wait_th(seq)
        lib.nvvl_free_sequence(seq)

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
        seq = lib.nvvl_create_sequence_device(count, self.device_id)

        self._set_process_desc(index_map=list(range(count)))

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
        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc(index_map=[0])

        if isinstance(indexs, int):
            indexs = [indexs]

        for index in indexs:
            self._start_receive(filename, index, 1)

        tensors = []
        for index in indexs:
            self._finish_reveive()
            t = self.tensor_queue.popleft()
            tensors.append(t["default"][0].cpu())

        return tensors

    def get_samples_old_sync(self, filename, indexs):
        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc(index_map=[0])

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

        seq = lib.nvvl_create_sequence_device(length, self.device_id)

        if not self.width or not self.height:
            image_shape = lib.nvvl_video_size_from_file(str.encode(filename))
            self.height = image_shape.height
            self.width = image_shape.width

        self._set_process_desc()

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


