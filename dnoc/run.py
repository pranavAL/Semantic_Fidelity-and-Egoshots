import argparse
import tensorflow as tf
import os
import json

from model import DNOC
from data import VisualTextData

tf.app.flags.DEFINE_string('stage', 'test',
                            """train/val/test""")

# Files settings
tf.app.flags.DEFINE_string('dataset_path', './prepare_data/mscoco/noc_coco_cap.json',
                           """dataset definition""")
tf.app.flags.DEFINE_string('cnn_dir', './prepare_data/mscoco/extracted_cnn_feature', """det result dir""")
tf.app.flags.DEFINE_string('det_dir', './prepare_data/mscoco/extracted_object_memory', """det result dir""")
tf.app.flags.DEFINE_string('save_dir', './logs/',
                           """checkpoints save dir""")
tf.app.flags.DEFINE_integer('max_queue_size', 100,
                            """input queue size""")
tf.app.flags.DEFINE_integer('num_loaders', 10,
                           """num thread of loading""")
# Model settings
tf.app.flags.DEFINE_integer('cnn_fea_dim', 4096, """cnn feature dim""")
tf.app.flags.DEFINE_integer('enc_cell_size', 1024, "encoder cell size")
tf.app.flags.DEFINE_integer('dec_cell_size', 1024, "decoder cell size")
tf.app.flags.DEFINE_integer('dec_seq_len', 15, "decoder sequence length")
tf.app.flags.DEFINE_integer('word_embedding_size', 1024, "word embedding size")
# Train settings
tf.app.flags.DEFINE_integer('batch_size', 256,
                          """training batch size""")
tf.app.flags.DEFINE_float('lr', 0.001,
                            """learning rate""")
tf.app.flags.DEFINE_integer('max_epochs', 20,
                          """maximum training epochs""")
tf.app.flags.DEFINE_float('weight_decay', 0.00005,
                          """weight decay""")
tf.app.flags.DEFINE_integer('step_per_print', 20,
                            """print loss step""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                          """moving_average_decay""")
tf.app.flags.DEFINE_integer('max_gradient_norm', 5,
                          """max gradient norm""")
# Captioning word setting 
tf.app.flags.DEFINE_integer('PAD_ID', 0, "PAD")
tf.app.flags.DEFINE_integer('GO_ID', 1, "GO")
tf.app.flags.DEFINE_integer('EOS_ID', 2, "EOS")
tf.app.flags.DEFINE_integer('UNK_ID', 3, "UNK")

# Query params, corresponding to the pre-trained detection model
tf.app.flags.DEFINE_integer('detection_classes', 80, "number of detection classes")
tf.app.flags.DEFINE_integer('max_det_boxes', 4, "max_det_boxes")
tf.app.flags.DEFINE_integer('det_fea_dim', 1088, "det_fea_dim") # the detection feature is extracted by "prepare_data"

FLAGS = tf.app.flags.FLAGS
sess_config = tf.ConfigProto(); sess_config.gpu_options.allow_growth = True

tf.reset_default_graph()

def eval_one_model(sess, model, epoch):
  uniq_name = []

  coord = tf.train.Coordinator()
  data_loader = VisualTextData(coord)
  data_loader.start()
  model_ckpt_path = "./logs/model.ckpt-{}".format(epoch)
  variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
  print("restore model from {}".format(model_ckpt_path))
  saver.restore(sess, model_ckpt_path)
  pred_captions = {}
  steps_per_epoch = data_loader.steps_per_epoch()

  for idx in range(1):
    batch = data_loader.next()
    predictions = model.eval_step(sess, batch)
    for name, predict in zip(batch['names'], predictions):
      pred_captions[name] = predict
      if name not in uniq_name:
        uniq_name.append(name)
  caption = {}        
  for name in uniq_name:

    caption[name] = pred_captions[name]
    sentence = ' '.join([i for i in caption[name]])
    print(id,'\t',sentence)
    with open('dnoc_ego.txt', 'a') as fp:
        fp.write(name+'\n'+sentence+'\n')

  data_loader.terminate()
  return


def eval():
  data = VisualTextData(tf.train.Coordinator())
  num_total_vocab, num_lstm_vocabs = data.get_vocab()
  # build model 
  model = DNOC(num_total_vocab, num_lstm_vocabs)
  with tf.Session(config=sess_config) as sess:
    eval_one_model(sess, model, 5)
    
eval() 
