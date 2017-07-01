import os
from apps.image_similarity.engine import DataManager
from embedding.greedy_encoding import GreedyEncoder
from utils.utils import ImageTransformer
from alg.EKNN import EmbeddingkNN
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import shutil
from model_configs.ae import *



def InitApp():
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..', '..')
    app_folder = os.path.join(project_root, 'apps', 'image_similarity')
    db_folder = os.path.join(project_root, 'data')
    raw_db_folder = os.path.join(db_folder, 'images')

    bin_db_folder = os.path.join(raw_db_folder, 'mat')
    raw_data_paths = [
        os.path.join(raw_db_folder, 'men'),
        os.path.join(raw_db_folder, 'women'),
        # os.path.join(raw_db_folder, 'toy'),
    ]
    model_name = 'c_2000_1000_300'
    model_path = os.path.join(project_root, 'models', 'vision', model_name)

    config = {
        'bin_db_path': bin_db_folder,
        'output_shape': output_shape,
        'db_name': 'fashion50K_embedding',
        'raw_db_paths': raw_data_paths
    }
    # Initialize encoder
    encoder = GreedyEncoder()
    encoder.load(model_path, model_name)


    mode = ['avg', 'max', 'min'][1]
    print('mode:',mode)
    t = ImageTransformer()
    t.configure(output_shape=output_shape)
    t.register_encoder(encoder)
    dm = DataManager()
    dm.configure(config)
    dm.register_encoder(encoder)
    '''
        RUN `load_raw_data` if first time
    '''
    train = dm.load_raw_data(batch_size=5000, flatten=True)
    # train = dm.load_dataset()
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
    while True:
        try:
            for batch in t.transform_all(query_folder, grey_scale=False, flatten=True):
                query = batch
                # print(query)
                # print(query.shape)
                # possible improvement, change the average to weighted average, assign more weights to recent clicks
                print('mode:', mode)
                if mode == 'avg':
                    query = np.mean(query, axis=0)
                elif mode == 'max':
                    query = np.amax(query, axis=0)
                elif mode == 'min':
                    query = np.amin(query, axis=0)
                else:
                    raise Exception('unkown option {0}'.format(mode))
                distances, indices = EMB.predict(np.array([query]))  # predict

                # =================================
                # Make k-recommendations using kNN prediction
                # =================================
                print("Making k-recommendations for each user...")

                # backward mapping to map indices back to vectors, and check the euclidean distance, if the mapping is correct
                # the euclidean distance of top 1 should be 0
                for i, (index, distance) in enumerate(zip(indices, distances)):
                    print("{0}: indices={1}, score={2}".format(i, index, distance))
                    answer_file_list = [dm.get_file_name(x) for x in index]
                    # print(answer_file_list)
                    answer_vec = t.transform_many(answer_file_list, flatten=True)
                    answer_vec = np.concatenate((answer_vec, np.array([query])))
                    print(euclidean_distances(answer_vec))
                    for answer_file in answer_file_list:
                        shutil.copy(answer_file, os.path.join(answer_folder, str(dm.get_index(answer_file))+'.jpg'))
        except FileNotFoundError:
            print('No query file found, did you add query images?')
            pass
        c = input('Continue? \nType `q` to break')
        if c == 'q':
            break

if __name__ == "__main__":
    InitApp()