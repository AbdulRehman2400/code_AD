import numpy as np



def data_load(Class_1_path_train, Class_2_path_train , Class_1_path_val, Class_2_path_val ):
    
    Class_1_np_data_train = np.load(Class_1_path_train)
    Class_2_np_data_train = np.load(Class_2_path_train)

    total_samples_class1_train= Class_1_np_data_train.shape[0]
    total_samples_class2_train= Class_2_np_data_train.shape[0]

    # Create label variables for both sets
    label_class1_train = np.array([0] * total_samples_class1_train)
    label_class2_train = np.array([1] * total_samples_class2_train)


    X_train = np.concatenate([Class_1_np_data_train, Class_2_np_data_train], axis=0)
    y_train = np.concatenate([label_class1_train, label_class2_train], axis=0)





    Class_1_np_data_val = np.load(Class_1_path_val)
    Class_2_np_data_val = np.load(Class_2_path_val)


    total_samples_class1_val  = Class_1_np_data_val.shape[0]
    total_samples_class2_val  = Class_2_np_data_val.shape[0]

    # Create label variables for both sets
    label_class1_val = np.array([0] * total_samples_class1_val)
    label_class2_val = np.array([1] * total_samples_class2_val)

    X_val = np.concatenate([Class_1_np_data_val, Class_2_np_data_val], axis=0)
    y_val = np.concatenate([label_class1_val, label_class2_val], axis=0)
    
    return X_train, y_train, X_val,y_val


