import os
from apps.image_similarity.engine import DataManager
from embedding.greedy_encoding import GreedyEncoder
from utils.utils import ImageTransformer
from alg.EKNN import EmbeddingkNN
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import shutil


'''
TODO @Charles
The indexing is wrong, because use the same picture as query won't return itself
'''


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
    train = dm.load_raw_data(batch_size=5000)
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
        for batch in t.transform_all(query_folder):
            query = batch
            # print(query)
            print(query.shape)
            centroid = np.mean(query, axis=0)
            distances, indices = EMB.predict(np.array([centroid]))  # predict

            # =================================
            # Make k-recommendations using kNN prediction
            # =================================
            print("Making k-recommendations for each user...")


            '''
            The file name to indices mapping is wrong, debug this

            Fix: don't load np array from file
            generate np array from raw file and build mappings
            '''

            # backward mapping to map indices back to vectors, and check the euclidean distance, if the mapping is correct
            # the euclidean distance of top 1 should be 0
            for i, (index, distance) in enumerate(zip(indices, distances)):
                print("{0}: indices={1}, score={2}".format(i, index, distance))
                answer_file_list = [dm.get_file_name(x) for x in index]
                print(answer_file_list)
                answer_vec= t.transform_many(answer_file_list)
                answer_vec = np.concatenate((answer_vec, np.array([centroid])))
                print(euclidean_distances(answer_vec))
                for answer_file in answer_file_list:
                    shutil.copy(answer_file, os.path.join(answer_folder, str(dm.get_index(answer_file))+'.jpg'))
        c = input('Continue? \nType `q` to break')
        if c == 'q':
            break

if __name__ == "__main__":
    pipeline()