TRAIN_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/training/*.tfrecord'
VALIDATION_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/validation/*.tfrecord'
TEST_DATA_PATTERN = 'gs://waymo_open_dataset_v_1_3_2/individual_files/testing/*.tfrecord'
NUM_CLASSES = 23

# Our baseline models.
POINTNET_MODEL_NAME = 'pointnet'
UNET_3D_MODEL_NAME = 'unet3d'
# CS348K project model.
PVCONV_MODEL_NAME = 'pvconv'
MODELS = [POINTNET_MODEL_NAME, UNET_3D_MODEL_NAME, PVCONV_MODEL_NAME]

