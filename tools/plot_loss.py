import os
import re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse(dir_event, tag):
    loss = []
    for e in tf.train.summary_iterator(os.path.join(model_path, dir_event)):
        for v in e.summary.value:
            if v.tag == tag:
                loss.append(v.simple_value)
    return loss

if __name__ == '__main__':
    model_path = '/netscratch/arenas/model/MovingSymbols2_NotSeen/'
    dir_event_FP = 'MovingSymbols2_NotSeen_FramePredictor_bce_mult/events.out.tfevents.1538582906.kasan'
    dir_event_FP_eval = 'MovingSymbols2_NotSeen_FramePredictor_bce_mult/eval/events.out.tfevents.1538583477.kasan'

    dir_event_DisFP = 'MovingSymbols2_NotSeen_DisentangledFP_bce_mult/events.out.tfevents.1538583130.kiew'
    dir_event_DisFP_eval = 'MovingSymbols2_NotSeen_DisentangledFP_bce_mult/eval/events.out.tfevents.1538583993.kiew'

    FP_loss = parse(dir_event_FP, 'loss_1')
    DisFP_loss = parse(dir_event_DisFP, 'loss_1')
    FP_loss_eval = parse(dir_event_FP_eval, 'loss')
    DisFP_loss_eval = parse(dir_event_DisFP_eval, 'loss')

    epochs = 100

    # Loss plot
    plt.plot(np.arange(0, epochs, float(epochs)/len(FP_loss)), FP_loss, color='C1', label='VanillaFP')
    plt.plot(np.arange(0, epochs, float(epochs)/len(FP_loss_eval)), FP_loss_eval, color='C1', ls='dashed', label='VanillaFP_val')
    plt.plot(np.arange(0, epochs, float(epochs)/len(DisFP_loss)), DisFP_loss, color='C2', label='DisentangledFP')
    plt.plot(np.arange(0, epochs, float(epochs)/len(DisFP_loss_eval)), DisFP_loss_eval, color='C2', ls='dashed', label='DisentangledFP_val')

    plt.ylabel('BCE loss')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.axis([0, 100, 0, 1])
    plt.legend(loc='upper right')
    plt.savefig('BCE_loss_NotSeen.png')
    plt.close()
