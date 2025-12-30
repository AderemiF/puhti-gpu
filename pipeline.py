import sys
import os
import pandas as pd
import numpy as np
import ollama

def prompt_on_image(image_path, model="ministral-3:3b"):
    prompt = "Describe briefly what happens in this image, in English."
    response = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_path]
    )
    return response['response']

def get_text_embedding(text, model="embeddinggemma"):
    response = ollama.embeddings(
        model=model,
        prompt=text
    )
    embedding = response['embedding']
    return np.array(embedding)

def create_model_df(filenames, descriptions, embeddings):
    return pd.DataFrame({
        "filename": filenames,
        "description": descriptions,
        "embedding": embeddings
    })

def load_model_csv(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x, sep=','))
        return df
    else:
        return pd.DataFrame(columns=["filename", "description", "embedding"])

def save_model_csv(model_df, filename):
    df_to_save = model_df.copy()
    df_to_save['embedding'] = df_to_save['embedding'].apply(lambda x: ','.join(map(str, x.tolist())))
    df_to_save.to_csv(filename, index=False)

def main(images_txt, model_csv):
    # Read image filenames from text file
    with open(images_txt, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]

    # Process images
    new_filenames = []
    new_descriptions = []
    new_embeddings = []
    frames_folder = 'frames'
    for img in image_files:
        print(f"Processing: {img}")
        desc = prompt_on_image(os.path.join(frames_folder, img))
        emb = get_text_embedding(desc)
        new_filenames.append(img)
        new_descriptions.append(desc)
        new_embeddings.append(emb)

    # Create DataFrame for new results
    new_df = create_model_df(new_filenames, new_descriptions, new_embeddings)

    # Load existing model and update
    model_df = load_model_csv(model_csv)
    updated_df = pd.concat([model_df, new_df], ignore_index=True)

    # Save updated model
    save_model_csv(updated_df, model_csv)
    print(f"Model updated and saved to {model_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_image_pipeline.py <images.txt> <model.csv>")
        sys.exit(1)
    images_txt = sys.argv[1]
    model_csv = sys.argv[2]
    main(images_txt, model_csv)