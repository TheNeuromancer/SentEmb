import tensorflow_hub as hub

# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/1")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

embeddings = embed(["The quick brown fox jumps over the lazy dog.", "I am a sentence for which I would like to get its embedding"])

print(embeddings)
# print(session.run(embeddings))

# The following are example embedding output of 512 dimensions per sentence
# Embedding for: The quick brown fox jumps over the lazy dog.
# [-0.016987282782793045, -0.008949815295636654, -0.0070627182722091675, ...]
# Embedding for: I am a sentence for which I would like to get its embedding.
# [0.03531332314014435, -0.025384284555912018, -0.007880025543272495, ...]