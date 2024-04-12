# Music Generation: Semantic User Input
Based on the parent paper Compose & Embellish: A Transformer-based Piano Generation System by Shih-Lun Wu and Yi-Hsuan Yang.  


## Prerequisites
  - **Python 3.8** and **CUDA 10.2** recommended
  - Install dependencies
    ```
    pip install -r requirements.txt
    pip install git+https://github.com/cifkao/fast-transformers.git@39e726864d1a279c9719d33a95868a4ea2fb5ac5
    ```
  - Download trained models from HuggingFace Hub (make sure you're in repository root directory)
    ```
    git clone https://huggingface.co/slseanwu/compose-and-embellish-pop1k7
    ```

[Download and listen to the parent paper output (audio file)](https://github.com/brianellis1997/Music_Generation/raw/main/Generated_Output/samp_01_2stage_samp01%20(1)%20-%20Chord-11_m.wav)

Here is a link to download one of our favorite user input specified outputs. The output is so odd because of the slow tempo and increase in temp (randomness).
The parameters specified were:
- `max_bars `: 77
- `temp`: 2.5
- `tempo`: 135
[Download and listen to the user input output (audio file)](https://github.com/brianellis1997/Music_Generation/raw/main/Generated_Output/User_Input_Generated.wav)

Have fun creating your own compositions!

## Acknowledgements
We would like to thank the following people for their open-source implementations that paved the way for our work:
  - [**Performer (fast-transformers)**](https://github.com/cifkao/fast-transformers/tree/39e726864d1a279c9719d33a95868a4ea2fb5ac5): Angelos Katharopoulos ([@angeloskath](https://github.com/angeloskath)) and Ondřej Cífka ([@cifkao](https://github.com/cifkao))
  - [**Transformer w/ relative positional encoding**](https://github.com/kimiyoung/transformer-xl): Zhilin Yang ([@kimiyoung](https://github.com/kimiyoung))
  - [**Musical structure analysis**](https://github.com/Dsqvival/hierarchical-structure-analysis): Shuqi Dai ([@Dsqvival](https://github.com/Dsqvival))
  - [**LakhMIDI melody identification**](https://github.com/gulnazaki/lyrics-melody/tree/main/pre-processing): Thomas Melistas ([@gulnazaki](https://github.com/gulnazaki))
  - [**Skyline melody extraction**](https://github.com/wazenmai/MIDI-BERT/tree/CP/melody_extraction/skyline): Wen-Yi Hsiao ([@wayne391](https://github.com/wayne391)) and Yi-Hui Chou ([@sophia1488](https://github.com/sophia1488))

## BibTex
If this repo helps with your research, please consider citing:
```
@inproceedings{wu2023compembellish,
  title={{Compose \& Embellish}: Well-Structured Piano Performance Generation via A Two-Stage Approach},
  author={Wu, Shih-Lun and Yang, Yi-Hsuan},
  booktitle={Proc. Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  url={https://arxiv.org/pdf/2209.08212.pdf}
}
```
