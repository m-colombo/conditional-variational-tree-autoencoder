import tensorflow as tf
from experiments.wmt_en_de import main as run_main, define_flags
import urllib.request
import json
import os

# FLAFS specific to the worker
tf.flags.DEFINE_string("worker_id",
                       default=None,
                       help='')

tf.flags.DEFINE_string("worker_type",
                       default=None,
                       help="CPU | GPU")

tf.flags.DEFINE_string("dispatcher_url",
                       default=None,
                       help='')

FLAGS = tf.flags.FLAGS


def main(argv):
    del argv

    if FLAGS.worker_type not in ["CPU", "GPU"] or FLAGS.worker_id is None:
        raise ValueError(FLAGS.worker_type)

    url = FLAGS.dispatcher_url+"?worker_id=%s&worker_type=%s" % (FLAGS.worker_id, FLAGS.worker_type)

    root_dir = FLAGS.model_dir
    original_batch_size = FLAGS.batch_size
    original_n_sample = FLAGS.n_sample

    with open('local_result_store.json', 'a+') as local_f:
        local_f.write('init\n')

        while True:

            if os.path.exists(FLAGS.worker_id+'_job_info.json'):
                print("Recovering...")
                with open(FLAGS.worker_id+'_job_info.json') as f:
                    job_info = json.load(f)
                    job_info_s = json.dumps(job_info)

            else:
                print("Querying dispatcher...")
                job_info_s = urllib.request.urlopen(url).read()

                if len(job_info_s) == 0:
                    print("no more model")
                    break

                job_info = json.loads(str(job_info_s, encoding='utf-8'))
                with open(FLAGS.worker_id+'_job_info.json', 'w') as f:
                    json.dump(job_info, f)

            # Overwrite tf flags
            for f in job_info['pars']:
                setattr(FLAGS, f, job_info['pars'][f])

            FLAGS.model_dir = os.path.join(root_dir, "model_"+str(job_info['id'])+"/")
            msg = {
                'model_id': job_info['id']
            }

            # launch
            print("\033[92mRunning " + str(job_info_s) + "'\033[0m")
            try:
                FLAGS.batch_size = original_batch_size
                FLAGS.n_sample = original_n_sample
                tf.assign(tf.train.get_or_create_global_step(), 0)
                msg['result'] = run_main(None)
                msg['status'] = 'success'

                if os.path.exists(FLAGS.worker_id + '_job_info.json'):
                    os.remove(FLAGS.worker_id + '_job_info.json')

            except Exception as e:
                msg['status'] = 'failed'
                msg['exception'] = str(e)
                print(e)

            # sending results:
            msg = json.dumps(msg)
            local_f.write(msg+'\n')
            urllib.request.urlopen(url, data=bytearray(msg, encoding='utf-8'))

        a = input('waiting to quit')

if __name__ == "__main__":
    define_flags()

    config = tf.ConfigProto()
    tf.enable_eager_execution(config=config)
    tf.contrib.eager.run()