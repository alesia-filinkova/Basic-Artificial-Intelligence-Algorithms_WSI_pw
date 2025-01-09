import kagglehub

# Download latest version
path = kagglehub.dataset_download("mrayushagrawal/us-crime-dataset")

print("Path to dataset files:", path)