# Synthtic Data Generator

This is the synthetic data generator based on [SynthTiger](https://github.com/clovaai/synthtiger).

Install `synthtiger`
```bash
pip install synthtiger
```

Generate synthtic data by:
```bash
synthtiger -o $outdir -w 8 synth_template.py SynthForCharDiffusion $config_file --count $max_num_of_samples
```

NOTICE: Please download background images from https://github.com/ankush-me/SynthText and put them in `ocr-dataset/SynthText/bg_data` and fonts data from google fonts and put them in `synthgenerator/resources/100fonts`.