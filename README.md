# RP_extract

Rhythm Pattern Audio Feature Extractor
for Music Similarity, Music Classification and Music Recommendation

created by:

TU Wien
Music Information Retrieval Group
Institute of Software Technology and Interactive Systems
Vienna University of Technology, Austria

Main Authors: Thomas Lidy (audiofeature), Alexander Schindler (slychief)


## Easy Getting Started

Analyze all audio files in a folder and store the extracted features:

```
python rp_extract_batch.py <input_path> <feature_file_name>
```

This will
- search for MP3 or WAV files in input_path
- extract a standard set of audio features (RP, SSD, RH - see http://ifs.tuwien.ac.at/mir/audiofeatureextraction.html )
- write them in a CSV like manner to feature_file_name (one for each feature type: .rp, .ssd, .rh)

Optionally specify the type of features you want:

```
python rp_extract_batch.py -rp -mvd -tssd <input_path> <feature_file_name>
```

Use the following to get a list of possible feature options:

```
python rp_extract_batch.py -h
```

## Use RP_extract in your code

rp_extract.py is the main feature extractor.
It can be imported and used in your code to do segment-level audio feature analysis.

For a step-by-step tutorial open „RP_extract Tutorial.ipynb“ in iPython Notebook or view the tutorial here:

http://nbviewer.ipython.org/github/tuwien-musicir/rp_extract/blob/master/RP_extract_Tutorial.ipynb


## More Information

http://ifs.tuwien.ac.at/mir
http://ifs.tuwien.ac.at/mir/downloads.html