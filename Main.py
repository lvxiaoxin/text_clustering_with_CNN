import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from keras.models import Model
from Utils import draw_clusters, calculate_similarity_matrixs, binarize
from Tokenization import EMBEDDING_DIM, get_corpus_core
from CNN import CNN

if __name__ == "__main__":

    # Step 1 - Corpus read, preprocess, and embeding calculation (X, Y)
    core_info = get_corpus_core()

    # Step 2 - Convert Y to B, the compact binary code
    B = binarize(core_info["Y"])
    TARGET_DIM = B.shape[1] # the last layer shape of CNN
    print("The shape of B:", B.shape)
    print("The shape of B[0]", B[0].shape)

    # Step 3 - Fix into CNN model and train
    model = CNN(
        MAX_NB_WORDS=core_info["MAX_NB_WORDS"], 
        EMBEDDING_DIM=EMBEDDING_DIM, 
        embedding_matrix=core_info["embedding_matrix"], 
        MAX_SEQUENCE_LENGTH=core_info["MAX_SEQUENCE_LENGTH"], 
        TARGET_DIM=TARGET_DIM, 
        EMBEDDING_TRAINABLE=False)
    model.fit(core_info["X"], B,  validation_split=0.2, epochs=50, batch_size=100, verbose=1, shuffle=True)

    # Step 4 - Extract another model which return the penultimate layer
    input = model.layers[0].input
    output = model.layers[-2].output
    model_penultimate = Model(input, output)

    H = model_penultimate.predict(core_info["X"])
    print("The target deep features representation shape: {}".format(H.shape))

    # Step 5 - Fit into KMeans
    Labels = core_info["labels"]
    n_clusters = len(np.unique(Labels))
    print("Numbers of clusters in label: %d" % n_clusters)
    normalize_H = normalize(H, norm='l2')

    sklearn_kmeans = KMeans(n_clusters=n_clusters).fit(normalize_H)

    # draw_clusters(normalize_H, sklearn_kmeans.labels_, sklearn_kmeans.cluster_centers_, axios=0.1)

    # Step 6 - Calculate the cosine similarity among the 18 clusters' center point
    similarities = calculate_similarity_matrixs(sklearn_kmeans.cluster_centers_)
    print("Similarities among clusters centers:")
    print(np.array(similarities))