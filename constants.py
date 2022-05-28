TRAIN_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/training/*.tfrecord'
VALIDATION_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/validation/*.tfrecord'
TEST_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/testing/*.tfrecord'
NUM_CLASSES = 23

X_MIN = -75.0
X_MAX = 77.0
Y_MIN = -75.0
Y_MAX = 77.0
Z_MIN = -6.0
Z_MAX = 7.0

# Our baseline models.
POINTNET_MODEL_NAME = 'pointnet'
# CS348K project models.
PVCONV_MODEL_NAME = 'pvconv'
PVCONV_SPARSE_BASIC_MODEL_NAME = 'pvconv_sparse'
PVCONV_SPARSE_SYMM_MODEL_NAME = 'pvconv_sparse_symm'
MODELS = [
    POINTNET_MODEL_NAME,
    PVCONV_MODEL_NAME,
    PVCONV_SPARSE_BASIC_MODEL_NAME,
    PVCONV_SPARSE_SYMM_MODEL_NAME
]

PVCONV_VOXEL_SIZE = 1.0

