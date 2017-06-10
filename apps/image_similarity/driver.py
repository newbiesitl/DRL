import os
from apps.image_similarity.engine import DataManager
from embedding.greedy_encoding import GreedyEncoder
from utils.utils import ImageTransformer
from alg.EKNN import EmbeddingkNN
import numpy as np
import shutil

def pipeline():
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..', '..')
    app_folder = os.path.join(project_root, 'apps', 'image_similarity')
    db_folder = os.path.join(app_folder, 'db')
    raw_db_folder = os.path.join(db_folder, 'img')

    bin_db_folder = os.path.join(db_folder, 'mat')
    raw_data_paths = [
        os.path.join(raw_db_folder, 'men'),
        os.path.join(raw_db_folder, 'women')
    ]
    model_name = '4000_2000_1000'
    model_path = os.path.join(app_folder, 'models', model_name)

    config = {
        'bin_db_path': bin_db_folder,
        'output_shape': (60,40),
        'db_name': 'fashion50K_embedding',
        'raw_db_paths': raw_data_paths
    }
    # Initialize encoder
    encoder = GreedyEncoder()
    encoder.load(model_path, '_'.join(['c', model_name]))


    t = ImageTransformer()
    t.configure(output_shape=(60,40))
    t.register_encoder(encoder)
    dm = DataManager()
    dm.configure(config)
    dm.register_encoder(encoder)
    '''
        RUN `load_raw_data` if first time
    '''
    # dm.load_raw_data(batch_size=5000)
    train = dm.load_dataset()
    dm.build_mapping()

    # =================================
    # Perform kNN
    # =================================
    print("Performing kNN to locate nearby items to user centroid points...")
    n_neighbors = 3  # number of nearest neighbours
    metric = "euclidean"  # kNN metric
    algorithm = "ball_tree"  # search algorithm

    EMB = EmbeddingkNN()  # initialize embedding kNN class
    # Compile and train model
    EMB.compile(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)  # compile kNN model
    EMB.fit(train)  # fit kNN


    # read from query folder
    query_folder = os.path.join(app_folder, 'query')
    answer_folder = os.path.join(app_folder, 'answer')

    for batch in t.transform_all(query_folder):
        query = batch
        print(query)
        print(query.shape)
        centroid = np.mean(query, axis=0)
        distances, indices = EMB.predict(np.array([centroid]))  # predict

        # =================================
        # Make k-recommendations using kNN prediction
        # =================================
        answer_file_list = []
        print("Making k-recommendations for each user...")
        for i, (index, distance) in enumerate(zip(indices, distances)):
            print("{0}: indices={1}, score={2}".format(i, index, distance))
            answer_file_list.append(dm.idx_f(index))
        for answer_file in answer_file_list:
            shutil.copy(answer_file, os.path.join(answer_folder, str(dm.f_idx(answer_file))+'.jpg'))

if __name__ == "__main__":
    pipeline()