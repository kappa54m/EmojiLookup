# EmojiLookup CLI
A simple CLI tool to find emojis based on string matching and semantic similarity using fastText.

## Installation

### Development Mode
If you are within the project directory, you can run the program directly using `uv`:

```bash
# Install dependencies
uv sync

# Run program
uv run emojilookup
```

### Tool Installation
To install the program as a global tool in editable mode:

```bash
uv tool install --editable .

# Run
emojilookup
```

## Usage
Once started, the program provides an interactive prompt for queries.

Note: the underlying "database" is stored at: [./src/emojilookup/data/emoji.txt](./src/emojilookup/data/emoji.txt).
Source: https://github.com/LukeSmithxyz/voidrice/blob/master/.local/share/larbs/emoji

### Command Line Arguments
- `--fasttext-dim <int>`: (default: 300) Dimension of the fastText vectors. If a value less than 300 is provided, the model will be reduced. This reduces memory usage and speeds up vector operations.
- `--train`: If provided, the tool will train a small, local fastText model on the emoji descriptions instead of loading the pre-trained English model. This is faster to start but provides lower-quality semantic matching.

### Query Syntax
- **Basic Query**: Simply type a keyword (e.g., `smile`, `heart`, `pizza`).
- **Override Limits**: You can override the number of matches returned by suffixing your query with `:N,M`.
    - `N`: Number of top matches by string matching (default: 7).
    - `M`: Number of additional matches by fastText similarity (default: 5).
    - Example: `smile :11,4` returns the top 11 string matches and 4 semantic matches. These values persist until changed again.

### Features
- **String Matching**: Prioritizes exact word matches, prefixes, and fuzzy similarity based on the emoji list in `src/emojilookup/data/emoji.txt`.
- **FastText Similarity**: Uses a trained model to find semantically related emojis that might not share the same characters as your query.
- **Case Insensitive**: All queries and descriptions are processed in lowercase.

### Caveats
- Currently, subword matching does not work very well. For example, if you look up "smile", you will not find "smiling face" from string matching (but semantic match is more helpful in this case), but if you search "smil" you can find them.

## Development
- Improve searching/string matching
