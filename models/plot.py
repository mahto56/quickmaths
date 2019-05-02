from keras.models import load_model

from keras.utils.vis_utils import plot_model
model = load_model('/home/kayshu/OpenCVProjects/quickmaths/models/model_0.1v7.h5')
print(len(model.layers))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)