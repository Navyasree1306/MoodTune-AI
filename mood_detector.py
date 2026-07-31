"""
mood_detector.py
-----------------
MODULE MAPPING: Foundations of AI

Detects the user's mood from free-text input using a supervised ML
text classifier (TF-IDF + Logistic Regression) — a core "Foundations
of AI" technique: represent text numerically, train a model on labeled
examples, predict on new input with a confidence score.

Trained fresh each run on a small bundled labeled dataset (fast: <1s).
For a larger deployment you'd swap this for a fine-tuned transformer
or a bigger labeled corpus — see README.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MOODS = [
    "happy",
    "sad",
    "energetic",
    "calm",
    "angry",
    "romantic",
    "focused",
    "nostalgic",
    "stressed"
    "delulu"
]

# Labeled training set: (text example, mood label). More examples per class
# with varied phrasing so TF-IDF + Logistic Regression separates classes
# with more confidence than a bare 5-example-per-class set allowed.
TRAINING_DATA = [
    ("I feel amazing today everything is going great", "happy"),
    ("so excited and cheerful right now", "happy"),
    ("life is good I'm smiling a lot", "happy"),
    ("today was such a joyful fun day", "happy"),
    ("feeling great and full of positivity", "happy"),
    ("I am happy and things are going well", "happy"),
    ("feeling content and good today", "happy"),
    ("I'm in such a good mood right now", "happy"),
    ("had such a great day at college with friends", "happy"),
    ("feeling thankful and happy about how things turned out", "happy"),

    ("I feel really down and empty today", "sad"),
    ("everything feels heavy and I want to cry", "sad"),
    ("missing someone and feeling low", "sad"),
    ("I'm heartbroken and upset", "sad"),
    ("feeling lonely and blue right now", "sad"),
    ("I feel sad and down today", "sad"),
    ("feeling depressed and unmotivated", "sad"),
    ("everything feels sad and gloomy right now", "sad"),

    ("I need to hit the gym and pump myself up", "energetic"),
    ("feeling pumped and ready to go run", "energetic"),
    ("hyped up and full of energy right now", "energetic"),
    ("want something fast and intense to get moving", "energetic"),
    ("feeling powerful and want to dance hard", "energetic"),
    ("I have so much energy right now", "energetic"),
    ("feeling energetic and ready to workout", "energetic"),
    ("pumped up and ready to hit the gym", "energetic"),

    ("I just want to relax and unwind", "calm"),
    ("feeling peaceful and want something soothing", "calm"),
    ("need to chill out after a long day", "calm"),
    ("looking for something gentle and soft to relax to", "calm"),
    ("feeling mellow and at ease", "calm"),
    ("I am so relaxed and peaceful right now", "calm"),
    ("feeling calm and want to unwind quietly", "calm"),
    ("just want peace and quiet to relax", "calm"),

    ("I am so frustrated and annoyed right now", "angry"),
    ("feeling furious about what happened today", "angry"),
    ("everything is pissing me off", "angry"),
    ("I need to vent I'm really mad", "angry"),
    ("feeling irritated and on edge", "angry"),
    ("I am furious right now", "angry"),
    ("feeling angry and frustrated with everything", "angry"),
    ("so mad about what just happened", "angry"),

    ("thinking about my partner feeling all warm inside", "romantic"),
    ("feeling loved and in love today", "romantic"),
    ("want something sweet and tender to listen to", "romantic"),
    ("missing my crush feeling butterflies", "romantic"),
    ("feeling affectionate and dreamy about someone", "romantic"),
    ("thinking about someone special today", "romantic"),
    ("feeling romantic and in love tonight", "romantic"),
    ("head over heels in love right now", "romantic"),
    ("totally in love and can't stop smiling", "romantic"),
    ("in the mood for something sweet and romantic", "romantic"),

    ("I need to concentrate and study for exams", "focused"),
    ("trying to get into deep work mode", "focused"),
    ("need background music while coding", "focused"),
    ("preparing for placements need to focus hard", "focused"),
    ("want something that helps me concentrate", "focused"),
    ("need to focus and study for my exam", "focused"),
    ("trying to concentrate on studying right now", "focused"),
    ("need focus music for deep work", "focused"),

    ("feeling nostalgic about my childhood days", "nostalgic"),
    ("thinking about old memories and old friends", "nostalgic"),
    ("missing the good old college days", "nostalgic"),
    ("feeling sentimental about the past", "nostalgic"),
    ("reminiscing about how things used to be", "nostalgic"),
    ("feeling nostalgic about old memories today", "nostalgic"),
    ("thinking back on the good old days", "nostalgic"),
    ("feeling sentimental about old times", "nostalgic"),

    ("want something upbeat and fun to dance to", "upbeat"),
    ("feeling good and want some feel good pop music", "upbeat"),
    ("in a good mood want something bouncy and fun", "upbeat"),
    ("need something upbeat to get the party going", "upbeat"),
    ("feeling bubbly and want a fun catchy song", "upbeat"),
    ("want something lively and upbeat right now", "upbeat"),
    ("feeling fun and want a bouncy feel good track", "upbeat"),
    ("want catchy upbeat music to lift my mood", "upbeat"),

    ("feeling delulu today just manifesting my dream life", "delulu"),
    ("in my delulu era thinking about my future husband", "delulu"),
    ("living in a fantasy world today feeling so delulu", "delulu"),
    ("manifesting my crush texting me first delulu vibes", "delulu"),
    ("feeling like the main character today so delulu", "delulu"),
    ("in my fantasy era imagining my dream life", "delulu"),
    ("feeling dreamy and delusional in a fun way", "delulu"),
    ("convinced something amazing is about to happen delulu energy", "delulu"),
    ("I'm stressed because of exams", "stressed"),
    ("Feeling overwhelmed with assignments", "stressed"),
     ("I have too much work to finish", "stressed"),
    ("I'm anxious about tomorrow's exam", "stressed"),
    ("I'm worried about my interview", "stressed"),
     ("I'm mentally exhausted", "stressed"),
   ("Too much pressure lately", "stressed"),
   ("Everything feels overwhelming", "stressed"),
   ("I'm burned out from studying", "stressed"),
    ("Feeling anxious and stressed today", "stressed"),
]


class MoodDetector:
    def __init__(self):
        texts = [t for t, _ in TRAINING_DATA]
        labels = [m for _, m in TRAINING_DATA]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=2000, C=8.0)
        self.model.fit(X, labels)

    def detect(self, text: str):
        """Returns (top_mood, confidence, full_probability_dict)."""
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        prob_dict = {str(c): round(float(p), 3) for c, p in zip(classes, probs)}
        top_mood = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_mood]
        return top_mood, confidence, prob_dict
