import os,getopt
import sys
sys.path.insert(0, "../")
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR,format='%(asctime)s %(name)s  %(levelname)s %(message)s')

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "g:b:d:m:u:e:", ["gpu=","batch_size=","data=","mode=","univar=","epochs="])
    except:
        print(u'input error!')
        sys.exit(2)
    
    gpu = None
    batch_size = None
    epochs = 30
    data = None
    mode = None
    univar = False
    
    for opt, arg in opts:
        if opt in ['-g','--gpu']:
            gpu = arg
        elif opt in ['-b','--batch_size']:
            batch_size = int(arg)
        elif opt in ['-d','--data']:
            data = arg
        elif opt in ['-m','--mode']:
            mode = arg
        elif opt in ['-u','--univar']:
            univar = bool(int(arg))
        elif opt in ['-e', '--epochs']:
            epochs = int(arg)
        else:
            print(u'input error!')
            sys.exit(2)
    
    if gpu is None or batch_size is None:
        print(u'input error!')
        sys.exit(2)
        
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    from src.utils import dataset_loader
    from src.core.model import GTT,ModelConfig,TSFoundation
    import tensorflow as tf
    
    input_len = 1024
    pred_len = 720
    if data == 'h1':
        train_df,val_df,test_df,signals = dataset_loader.load_ett_data(name='ETTh1.csv')
    elif data == 'h2':
        train_df,val_df,test_df,signals = dataset_loader.load_ett_data(name='ETTh2.csv')
    elif data == 'm1':
        train_df,val_df,test_df,signals = dataset_loader.load_ett_data(name='ETTm1.csv')
    elif data == 'm2':
        train_df,val_df,test_df,signals = dataset_loader.load_ett_data(name='ETTm2.csv')
    elif data == 'electricity':
        train_df,val_df,test_df,signals = dataset_loader.load_electricity_data()
    elif data == 'traffic':
        train_df,val_df,test_df,signals = dataset_loader.load_traffic_data()
    elif data == 'weather':
        train_df,val_df,test_df,signals = dataset_loader.load_weather_data()
    elif data == 'ill':
        train_df,val_df,test_df,signals = dataset_loader.load_illness_data()
        input_len = 128
        pred_len = 60
    
    foundation_path= f'../checkpoints/GTT-{mode}'
    pm = TSFoundation.load_model(foundation_path)
    
    cp=f'../checkpoints/GTT-finetune'
    mc = ModelConfig(block_size=pm.configs.block_size,
                     patch_size=pm.configs.patch_size,
                     pred_len = pred_len,
                     enable_revin = True,
                     affine = True,
                     revin_time = True)
    model = GTT(signals,mc)
    hist = model.train(train_df, val_df, cp, pm=pm, batch_size=batch_size, epochs=epochs, distribute=True, verbose=1)
    model.save_model(cp) 
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = model.load_model(cp)
        y_pred,y_true = model.predict_ft(test_df,input_len,pred_len,batch_size=batch_size*4)
    
    if pred_len == 720:
        pred_lens = [96,192,336,720]
    else:
        pred_lens = [24,36,48,60]
    
    maes = []
    mses = []
    for pred_len in pred_lens:
        mae = np.mean(np.abs(y_pred[:,:pred_len,:]-y_true[:,:pred_len,:]))
        mse = np.mean(np.square(y_pred[:,:pred_len,:]-y_true[:,:pred_len,:]))
        print('mae', mae)
        print('mse', mse)
        maes.append(mae)
        mses.append(mse)
    print('mae mean',np.mean(maes))
    print('mse mean',np.mean(mses))
    print()
    
    