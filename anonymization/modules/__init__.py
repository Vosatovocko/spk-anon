from .tts import SpeechSynthesis
# Commented out as Madhumitha said
#from .text import SpeechRecognition
from .prosody import ProsodyExtraction, ProsodyAnonymization
from .speaker_embeddings import SpeakerExtraction, SpeakerAnonymization
# To solve path error as suggested by Nha:
from .speech_embeddings.speech_recognition import SpeechRecognitionSFM
