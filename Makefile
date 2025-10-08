prep:
	 uv run python3 src/music_clustering/01_extract_features.py
train:
	 uv run python3 src/music_clustering/02_clustering.py
