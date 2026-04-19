import re
import difflib
import fasttext
import os
import sys
import numpy as np
import argparse
import importlib.resources as resources

# Import readline to enable command history (Up arrow, Ctrl+P) and line editing
try:
    import readline
except ImportError:
    # readline is not available on some platforms (like Windows without pyreadline)
    pass

def load_emojis(file_path):
    emojis = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                emoji = parts[0]
                description = parts[1].lower()
                emojis.append((emoji, description))
    return emojis

def train_fasttext_model(emojis, dim=300, model_path="emoji_model.bin"):
    print("Training fasttext model (dim: {}, target: {})".format(dim, model_path))
    # Prepare data for training
    with open("temp_train.txt", "w", encoding="utf-8") as f:
        for _, description in emojis:
            f.write(description + "\n")
    
    # Train unsupervised skipgram model
    model = fasttext.train_unsupervised("temp_train.txt", model='skipgram', dim=dim, minn=2, maxn=5, epoch=50)
    model.save_model(model_path)
    os.remove("temp_train.txt")
    return model

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)

class EmojiLookup:
    def __init__(self, data_path, fasttext_dim=300, do_train=False):
        self.emojis = load_emojis(data_path)
        self.fasttext_dim = fasttext_dim
        self.n = 7
        self.m = 5
        self.model = self._load_fasttext_model(do_train=do_train)
        
        # Precompute vectors for emoji descriptions
        self.emoji_vectors = [self.model.get_sentence_vector(desc) for _, desc in self.emojis]

    def _load_fasttext_model(self, do_train: bool):
        if not do_train:
            import fasttext.util
            print("Ensuring pre-trained fastText model (English) is available...")
            fasttext.util.download_model('en', if_exists='ignore')
            og_model_pth = 'cc.en.300.bin'
            
            if self.fasttext_dim >= 300:
                print("Loading original 300-d fastText model from '{}'".format(og_model_pth))
                return fasttext.load_model(og_model_pth)
                
            reduced_model_path = f'cc.en.{self.fasttext_dim}.bin'
            if not os.path.isfile(reduced_model_path):
                print("Loading original 300-d fastText model from '{}' for reduction".format(og_model_pth))
                ft = fasttext.load_model(og_model_pth)
                print("Reducing model dimension to: {}".format(self.fasttext_dim))
                fasttext.util.reduce_model(ft, self.fasttext_dim)
                ft.save_model(reduced_model_path)
                return ft
            else:
                print("Loading reduced fastText model from '{}'".format(reduced_model_path))
                return fasttext.load_model(reduced_model_path)
        else:
            return train_fasttext_model(self.emojis, dim=self.fasttext_dim)

    def search(self, query):
        query = query.lower()
        
        # 1. String matching similarity
        scores_string = []
        for i, (emoji, description) in enumerate(self.emojis):
            desc_words = description.split()
            
            max_word_score = 0
            for word in desc_words:
                if query == word:
                    max_word_score = max(max_word_score, 1.0)
                elif word.startswith(query):
                    max_word_score = max(max_word_score, 0.9)
                elif query in word:
                    max_word_score = max(max_word_score, 0.75)
            
            if max_word_score == 0:
                # Fallback to difflib
                desc_words = description.split(" ")
                for desc_word in desc_words:
                    max_word_score += difflib.SequenceMatcher(None, query, desc_word).ratio() * 0.7 / len(desc_words)
                
            scores_string.append((max_word_score, i))
        
        scores_string.sort(key=lambda x: (x[0], -len(self.emojis[x[1]][1])), reverse=True)
        top_n_indices = [idx for score, idx in scores_string[:self.n]]
        top_n_results = [(self.emojis[idx][0], self.emojis[idx][1]) for idx in top_n_indices]
        
        # 2. FastText similarity for remaining matches
        query_vec = self.model.get_sentence_vector(query)
        scores_fasttext = []
        seen_indices = set(top_n_indices)
        
        for i, (emoji, description) in enumerate(self.emojis):
            if i in seen_indices:
                continue
            
            vec = self.emoji_vectors[i]
            score = cosine_similarity(query_vec, vec)
            scores_fasttext.append((score, i))
            
        scores_fasttext.sort(key=lambda x: x[0], reverse=True)
        top_m_indices = [idx for score, idx in scores_fasttext[:self.m]]
        top_m_results = [(self.emojis[idx][0], self.emojis[idx][1]) for idx in top_m_indices]
        
        return top_n_results, top_m_results

    def run(self):
        print("Emoji Lookup CLI")
        print("Type your query or 'exit' to quit.")
        print("Format: query :N,M to override counts.")
        
        pattern = re.compile(r"^(.*?)\s+:(\d+),(\d+)$")
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                if not user_input or user_input.lower() == "exit":
                    break
                
                match = pattern.match(user_input)
                if match:
                    query_text = match.group(1).strip()
                    self.n = int(match.group(2))
                    self.m = int(match.group(3))
                else:
                    query_text = user_input
                
                top_n, top_m = self.search(query_text)
                
                print(f"\nTop {len(top_n)} matches by string matching:")
                for emoji, desc in top_n:
                    print(f"{emoji} {desc}")
                
                if top_m:
                    print(f"\nTop {len(top_m)} additional matches by fastText similarity (dim={self.fasttext_dim}):")
                    for emoji, desc in top_m:
                        print(f"{emoji} {desc}")
                        
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Emoji Lookup CLI")
    parser.add_argument("--fasttext-dim", type=int, default=300, 
                        help="Dimension of fastText vectors. If < 300, the model will be reduced. (default: 300)")
    parser.add_argument("--train", "-t", action="store_true",
                        help="Train a fastText model on the emoji descriptions locally instead of loading a pre-trained model.")
    args = parser.parse_args()

    try:
        data_path = resources.files("emojilookup.data").joinpath("emoji.txt")
        if not data_path.is_file():
            print(f"Error: Emoji data file not found at {data_path}", file=sys.stderr)
            sys.exit(1)
            
        lookup = EmojiLookup(str(data_path), fasttext_dim=args.fasttext_dim, do_train=args.train)
        lookup.run()
    except Exception as e:
        print(f"Critical Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
