from .dataset import VideoDataset,ProcessDesc
from .loader import VideoLoader
from .video_reader import VideoReader, PreProcess

def video_size_from_file(filename):
    return lib.nvvl_video_size_from_file(str.encode(filename))

def video_frame_count_from_file(filename):
    return lib.nvvl_video_frame_count_from_file(str.encode(filename))
