# RP_extract:
## Rhythm Pattern Audio Feature Extractor
for Music Similarity, Music Classification and Music Recommendation

created by:

TU Wien<br>
[Music Information Retrieval Group](http://ifs.tuwien.ac.at/mir)<br>
Institute of Software Technology and Interactive Systems<br>
Vienna University of Technology, Austria

RP_extract is a Python library to extract semantic features (so called audio descriptors) from audio files (WAV, MP3, ...)
which can be used in tasks such as finding similar sounding music, creating playlists or recommender systems,
categorizing music into a custom set of categories such as genres, and detecting concepts such as moods and emotions in music.
Most of these tasks are achieved through employing Machine Learning, example implementations are provided in this
repository and the tutorials included.

Main Authors: Thomas Lidy (audiofeature), Alexander Schindler (slychief)

## Installation

[Python 2.7] (https://www.python.org/downloads/release/python-2712/) is required.

### Operating Systems

Linux, Mac and Windows are supported. We recommend Ubuntu 14.04 or 16.04.

### Download

Either download as ZIP from https://github.com/tuwien-musicir/rp_extract/archive/master.zip , or:

```
git clone https://github.com/tuwien-musicir/rp_extract.git
```

### Install Dependencies

On Linux Ubuntu many dependencies can be installed from the Software Center or repository like this:

```
sudo apt-get install python-numpy python-scipy python-pandas python-scikits-learn
```

Use this to install the remaining dependencies on Ubuntu, respectively all dependencies on Mac and Windows:

```
sudo pip install -r requirements.txt
```

Note that some of the requirements are only needed for specific parts of the library. If all you want to use is the
audio analysis part with `rp_extract.py`, `numpy` and `scipy` are the only requirements.

### Optional Dependencies

#### MP3 Decoder

If you want to use MP3, M4A, FLAC or AIF(F) files as input, you need to have one of the following decoders installed in your system:
(Note: lame and mpg123 only support MP3, ffmpeg supports MP3 and all other formats)

- Linux: install `ffmpeg`, `mpg123`, or `lame` from your Software Install Center or package repository (via `sudo apt-get install`)
  - install ffmpeg on Ubuntu 14.04: see http://fcorti.com/2014/04/22/ffmpeg-ubuntu-14-04-lts
  - install ffmpeg on Debian Jessie:
    `sudo echo deb http://ftp.uk.debian.org/debian jessie-backports main > /etc/sources.list.d/ffmpeg.list;
    sudo apt-get update;
    sudo apt-get install ffmpeg`
- Mac: FFMPeg for Mac: http://ffmpegmac.net or Lame for Mac: http://www.thalictrum.com/en/products/lame.html
- Windows: FFMpeg.exe is already included (nothing to install)

#### Other

For plotting (using `rp_plot.py`)
```
sudo apt-get install python-matplotlib
```

For HDF5 file output instead of CSV:

Ubuntu Linux:
```
sudo apt-get install libhdf5-serial-dev python-tables
```

Mac OS X:
([homebrew](http://brew.sh) must be installed first)
```
brew tap homebrew/science
brew install hdf5
sudo pip install tables
```

## Easy Getting Started

Analyze all audio files in a folder and store the extracted features:

```
python rp_extract_batch.py <input_path> <feature_file_name>
```

This will
- search for WAV, MP3, M4A, FLAC or AIFF files in `input_path`
- extract a standard set of audio features (RP, SSD, RH - see http://ifs.tuwien.ac.at/mir/audiofeatureextraction.html )
- write them in CSV format to `feature_file_name` (don't specify a file extension, it will create 3 files, one for each feature type: .rp, .ssd, .rh)

Optionally specify the type of features you want:

```
python rp_extract_batch.py -rp -mvd -tssd <input_path> <feature_file_name>
```

Use the following to get a list of possible feature options:

```
python rp_extract_batch.py -h
```

## Use RP_extract in your code

`rp_extract.py` is the main feature extractor.
It can be imported and used in your code to do segment-level audio feature analysis.

For a step-by-step tutorial open „RP_extract Tutorial.ipynb“ in iPython Notebook or view the tutorial here:

http://nbviewer.ipython.org/github/tuwien-musicir/rp_extract/blob/master/RP_extract_Tutorial.ipynb

It also includes examples on how to compute music similarity, e.g. for music recommendation or
creating playlists of coherent music.

## Genre Recognition and Classification

`rp_classify.py` will analyze audio files and categorize them into a high-level concept (such as genre, style or mood)
given a pre-trained classifier model that was created based on training data.

It can be used on a single audio file (wav or mp3) to recognize its genre like this:

```
python rp_classify.py music/BoxCat_Games_-_10_-_Epic_Song.mp3
```

will output:

```
music/BoxCat_Games_-_10_-_Epic_Song.mp3:	pop
```

You can also use a folder as an input, to analyze and predict all audio files contained. The full syntax is:

```
python rp_classify.py input_path model_file output_filename
```

`input_path` can be: a folder, wav, mp3, m4a or aif(f) file, or txt file containing a line-wise list of audio files

The pre-trained model included in this code repository (`models/GTZAN`) can predict these 10 genres:

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

`output_filename` is an optional file in which predictions will be written in a TAB-separated way
(if omitted, result will be printed to stdout).


### Train a model:

You can train your own model like this:

```
python rp_classify.py -t input_path model_file
```

In this case files from `input_path` will be read and analyzed and a model will be trained and stored in `model_file`.

In this default case, files in `input_path` _must_ be organized in sub-folders named like the categories to be used for training (e.g. one folder named 'pop', one for 'electronic', one for 'classical' etc.)

Alternatively, you can provide the class labels for training a new model in two alternative ways, adding an additional parameter to the command line:

* `-c classfile`: expects the name of a TAB-separated file, where each line contains `<audiofilename>` `TAB` `<class_label>`
* `-m multiclassfile`: also a TAB-separated file with `<audiofilename>` in the first column, and additional columns with the class labels in the header (1st line) and an 'x' for each file belonging to the class, an empty TAB position otherwise.

Once you trained a model in one of these ways, you can do predictions like above:

```
python rp_classify.py input_path model_file output_filename
```

Note: Specify the `model_file` always without file extension as 3 files will be generated with different extensions.


## More Information

http://ifs.tuwien.ac.at/mir

http://ifs.tuwien.ac.at/mir/audiofeatureextraction.html

http://ifs.tuwien.ac.at/mir/downloads.html
