Task 3 — Music Generation with AI

What it does

A deep learning system trained on Johann Sebastian Bach's chorales that generates original classical-style piano music. The model learns musical patterns from real compositions and produces new pieces it has never played before.
Training data : 20 Bach chorales (bundled in music21 — no download)
Model input   : 100 consecutive notes
Model output  : predicted next note (from vocabulary of ~130 unique notes)
Generation    : repeat 200 times → full composition
Audio output  : WAV file via FluidSynth piano samples
Features

Trained on Bach chorales from music21's built-in corpus (zero external downloads)
Bidirectional LSTM — reads sequences both forward and backward
Temperature sampling for creativity control
Gradio UI with audio player and MIDI download
Training loss curve visualization
80-epoch training with EarlyStopping + ModelCheckpoint
Compatible with TensorFlow 2.20 (Colab Python 3.12)

Architecture
Input: sequence of 100 integer-encoded notes
    ↓
Embedding(n_vocab, 64)         → note integers → 64-dim learned vectors
    ↓
Bidirectional LSTM(256)        → reads forward + backward simultaneously
    ↓
Dropout(0.3)                   → prevents overfitting
    ↓
LSTM(256)                      → higher-level pattern learning
    ↓
Dropout(0.3)
    ↓
Dense(256, relu) + BatchNorm   → non-linear combination
    ↓
Dense(n_vocab, softmax)        → probability for each possible next note
    ↓
Temperature sampling           → pick next note based on creativity setting
Temperature Explained
pythonpreds = log(preds + ε) / temperature   # rescale log-probabilities
preds = softmax(preds)                  # re-normalize to valid distribution
idx   = multinomial_sample(preds)       # sample next note

# temperature = 0.3 → peaks sharpen   → safe, classical, repetitive
# temperature = 1.0 → standard        → balanced creativity
# temperature = 1.5 → peaks flatten   → adventurous, jazz-like
Pipeline
Bach chorales (music21 corpus)
        ↓
Note extraction: "C4", "E4", "0.4.7", "G4" ...
        ↓
Integer encoding + sliding window sequences (length=100)
        ↓
Bidirectional LSTM training (~20 min on T4 GPU)
        ↓
Temperature sampling → 200 generated notes
        ↓
music21 Stream → .mid file
        ↓
fluidsynth + FluidR3_GM.sf2 → .wav audio
        ↓
Gradio audio player + MIDI download
Stack
LibraryPurposemusic21 9.1.0MIDI parsing, note extraction, score writingtensorflow 2.20Bidirectional LSTM model + trainingnumpySequence arrays and temperature mathfluidsynthRender MIDI → WAV with real piano samplesgradio 3.50.2Web interface with audio player
Run
bash# Enable GPU in Colab first: Runtime → Change Runtime Type → T4 GPU
apt-get install -y fluidsynth fluid-soundfont-gm
pip install music21==9.1.0 gradio==3.50.2

python Task3_MusicGeneration/Task3_Music_Generation_FIXED.py

Training time: ~20 minutes on T4 GPU. Run all cells top to bottom.
After Cell 6 finishes, the Gradio UI launches and you can generate music immediately.


Tech Stack
Language      Python 3.12
Environment   Google Colab (T4 GPU)
Interface     Gradio 3.50.2

Task 1        deep_translator · langdetect
Task 2        nltk · scikit-learn · pandas · numpy
Task 3        tensorflow 2.20 · music21 · fluidsynth
All tasks     gradio · numpy · matplotlib

How to Run
Option A — Google Colab (recommended)

Go to colab.google.com → New Notebook
For Task 3: Runtime → Change Runtime Type → T4 GPU → Save
Paste the task file contents into code cells
Run cells top to bottom (Shift+Enter)
Copy the gradio.live public URL from the output

Option B — Local Machine
bashgit clone https://github.com/YOUR_USERNAME/codealpha_tasks.git
cd codealpha_tasks

# Task 1
pip install deep_translator langdetect gradio
python Task1_LanguageTranslation/Task1_Language_Translation_Tool.py

# Task 2
pip install nltk scikit-learn gradio pandas numpy
python Task2_FAQChatbot/Task2_FAQ_Chatbot_FIXED.py

# Task 3 (Linux/Mac — requires fluidsynth)
sudo apt-get install fluidsynth fluid-soundfont-gm
pip install music21==9.1.0 gradio==3.50.2
python Task3_MusicGeneration/Task3_Music_Generation_FIXED.py
Requirements by Task
RequirementTask 1Task 2Task 3GPU——RecommendedInternetRequired——RAM2 GB2 GB4 GB+Training timeInstant~30 sec~20 min

Project Structure
codealpha_tasks/
│
├── README.md
│
├── Task1_LanguageTranslation/
│   └── Task1_Language_Translation_Tool.py
│
├── Task2_FAQChatbot/
│   └── Task2_FAQ_Chatbot_FIXED.py
│
└── Task3_MusicGeneration/
    └── Task3_Music_Generation_FIXED.py

What I Learned
Task 1

How translation APIs abstract away language complexity
Statistical language detection using character frequency models
Building production-grade Python web apps in under 100 lines with Gradio

Task 2

Full NLP preprocessing pipeline: why "running" and "run" should match
Why TF-IDF gives "algorithm" more weight than "the"
Why cosine similarity measures angle (meaning), not distance (exact words)
How real chatbots work — it's math, not magic

Task 3

How music can be represented as a sequence prediction problem
Why bidirectional LSTM outperforms standard LSTM for sequential patterns
How temperature rescales a probability distribution to control creativity
The engineering tradeoff: more training data = better music, but slower training
TF 2.20 breaking changes vs TF 2.13 (.keras format, Keras 3 API)


Acknowledgements

Johann Sebastian Bach — for the training data
CodeAlpha — for the internship structure and mentorship
music21 — for bundling 371 Bach chorales in a Python library
Ultralytics — for making YOLOv8 accessible to students


<div align="center">
Built during the CodeAlpha AI Internship — May 2026
Python · TensorFlow · music21 · NLTK · YOLOv8 · Gradio · OpenCV
</div>
