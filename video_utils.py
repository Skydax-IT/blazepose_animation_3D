import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
from datetime import datetime

########### CLASSES ###########

class VideoReader():
    """To read a video
    """
    def __init__(self, path: str, max_sec=0, transform=None, gray = False) -> None:
        """Constructor

        Parameters
        ----------
        path : str
            path to the video file
        max_sec : int, optional
            number of second maximum treated, by default 0
        transform : optional
            transformation to apply, by default None
        gray : bool, optional
            if the video is read as grayscale, by default False

        Raises
        ------
        NotFoundErr
            If the file is not found
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f'The video with path {path} can\'t be found')

        self. path = path
        self.video = cv2.VideoCapture(path)
        self.rate = self.video.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.video.get(3))
        self.height = int(self.video.get(4))
        self.id = 0
        self.max_frames = max_sec*self.rate
        self.transform = transform
        self.n = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gray = gray
        self.temp = None

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def shape(self):
        return (self.height, self.width, 3)

    @property
    def frame_rate(self) -> int:
        return self.rate

    def __len__(self):
        return max(self.n,self.max_frames) if self.max_frames > 0 else self.n
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.video.isOpened() or 0 < self.max_frames <= self.id or self.id >= self.n:
            self.video.set(cv2.CAP_PROP_POS_FRAMES,0)
            raise StopIteration
        self.id += 1
        valid, frame = self.video.read()
        while not valid:
            self.id += 1
            valid, frame = self.video.read()
            if not self.video.isOpened() or self.id >= self.n:
                self.video.set(cv2.CAP_PROP_POS_FRAMES,0)
                raise StopIteration
        
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
            
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


    def get_current_frame(self) -> np.ndarray:
        """return the current frame of the video without passing to the next frame

        Returns
        -------
        np.ndarray
            the frame returned
        """
        id_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        valid, frame = self.video.read()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, id_frame)
        return frame if valid else None


class VideoWriter:
    """Object that can write a video
    """
    def __init__(self, path : str, frame_rate : int, size : Tuple[int, int], gray : bool = False):
        """constructor

        Parameters
        ----------
        path : str
            path where the video will be written
        frame_rate : int
            frame rate of the video
        size : Tuple[int, int]
            size of the video
        gray : bool, optional
            if the video is in gray scale, by default False
        """
        self.gray = gray
        
        if os.path.isfile(path):
            raise FileExistsError(f'The video with path {path} already exists')
        
        if gray:
            self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, size,0)
        else:
            self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, size)
        self.rate = frame_rate
    
    def write_numpy(self, frame : np.ndarray):
        """write an image from a numpy array

        Parameters
        ----------
        frame : np.ndarray
            the frame to be written
        """
        if self.gray:
            frame = (frame*255).astype(np.uint8)
        self.video.write(frame)
    
    
    def write(self, frames):  # for tensor
        """write severals images from a torch tensor

        Parameters
        ----------
        frames : torch Tensor [T, C, H, W]
            the frames to be written
        """
        if frames.size(1) == 1:  # Si un seul channel i.e. in graysacale
            frames = frames.repeat(1, 3, 1, 1) # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            self.video.write(frame)
                
    def close(self):
        """finish the video
        """
        self.video.release()


class DataWriter():
    def __init__(self, path, columns : list, *args, **kwargs) -> None:
        self.path = path
        self.df = pd.DataFrame(columns=columns)
    
    def write(self, data):
        row = pd.DataFrame.from_dict(data)
        self.df = pd.concat([self.df,row],ignore_index=True)
    
    def close(self):
        self.df.to_csv(self.path, header=True, index=False)
        del self.df
        

########### FUNCTIONS ###########

def make_dirs(path : str, folders : List[str]) -> str:
    """create tangled folders from the path directory

    Parameters
    ----------
    path : str
        path to add the folder
    folder : list[str]
        name of the folders c

    Returns
    -------
    str
        the path of the folder
    """
    current_path = path
    for folder in folders:
        current_path = os.path.join(current_path,folder)
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
    return current_path


def get_out_folder(path : str, name : str = 'results') -> str:
    """create and return a path to store result from an input file of path `path`

    Parameters
    ----------
    path : str
        input path
    name : str, optional
        name of the folder result, by default 'results'

    Returns
    -------
    str
        the path to the created folder
    """
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    now = datetime.now()
    folder = f'{now.strftime("%Y.%d.%m %Hh%Mm%Ss")} {name}'
    return make_dirs(dirname, [basename, folder])


def get_out_path(path : str, dirname : str = None, add : str = 'out', ext : str = None) -> str:
    """Give a save output path in folder result based on the input path in parameter

    Parameters
    ----------
    path : str
        input path of the file
    dirname : str, optional
        precise the folder path if you have one (should be absolute path), by default None
    add : str, optional
        aggregate to add in the name of the file, by default 'out'
    ext : str, optional
        precise extantion if needed, by default None

    Returns
    -------
    str
        output path
    """
    if dirname is None:
        dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    start_path = os.path.join(dirname,basename)
    if ext is None:
        ext = os.path.splitext(path)[-1].lower()
        if ext in ['.mp4','.mov','.avi']:
            ext = '.mp4'
    out_path = f'{start_path}_{add}{ext}'
    i = 1
    while os.path.isfile(out_path):
        out_path = f'{start_path}_{add}_{str(i)}{ext}'
        i+=1
    return out_path


def get_image(path : str, to_RGB : bool = False, gray : bool = False) -> np.ndarray:
    """get an image from a path

    Parameters
    ----------
    path : str
        path to the file
    to_RGB : bool, optional
        if the image is converted in RGB (BGR by default), by default False
    gray : bool, optional
        if the image should by loaded as grayscale (not compatible with to_RGB), by default False

    Returns
    -------
    np.ndarray
        the image
    """
    assert not (to_RGB and gray), 'to_RGB and gray can\'t be set both as True'
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'\nThe file "{path}" can\'t be found.\n')
    
    if not gray:
        image = cv2.imread(path)
        if to_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else :
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def print_image(image : np.ndarray) -> None:
    """print image from a numpy BGR ndarray

    Parameters
    ----------
    image : np.ndarray
        the image
    """
    cv2.namedWindow('Image',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_is_video(path : str) -> bool:
    """tell if the file with path `path` is a video or not

    Parameters
    ----------
    path : str
        the path of the file

    Returns
    -------
    bool
        if the file is a video
    """
    ext = os.path.splitext(path)[-1].lower()
    return ext in ['.mp4','.mov','.avi']

def merge_alphas(alphas : List[np.ndarray]) -> np.ndarray:
    """merge the alphas into one by taking the maximum

    Parameters
    ----------
    alphas : List[np.ndarray]
        List of the alpha masks

    Returns
    -------
    np.ndarray
        the merged mask
    """
    rslt = np.zeros_like(alphas[0])
    for alpha in alphas:
        rslt = np.maximum(rslt, alpha)
    return rslt

if __name__ == '__main__':
    ...