"""Streaming training/evaluation for LeafLens; see README.md for commands."""
import argparse,json,random
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import DEFAULT_ROOT,iter_samples,ALIASES
from image_preprocessing import preprocess_rgb
SIZE=(224,224); SEED=42; DEFAULT_BATCH_SIZE=16
CROPS=tuple(sorted(set(ALIASES.values())))

def disease_split(labels,seed):
    ids=np.arange(len(labels)); counts=Counter(labels); rare={x for x,n in counts.items() if n<2}
    common=np.asarray([i for i,x in enumerate(labels) if x not in rare],dtype=np.int32); rare_ids=np.asarray([i for i,x in enumerate(labels) if x in rare],dtype=np.int32)
    if len(common)>=2:
        try: train_common,val=train_test_split(common,test_size=.2,random_state=seed,stratify=[labels[i] for i in common])
        except ValueError: train_common,val=train_test_split(common,test_size=.2,random_state=seed)
        train=np.concatenate((train_common,rare_ids))
    else: train,val=train_test_split(ids,test_size=.2,random_state=seed)
    if not len(val): raise SystemExit('Not enough images to create a validation split.')
    return train,val

def make_dataset(tf,paths,labels,batch_size=16,training=False,seed=SEED):
    ds=tf.data.Dataset.from_tensor_slices(([str(p) for p in paths],np.asarray(labels,dtype=np.int32)))
    if training: ds=ds.shuffle(min(len(paths),max(512,batch_size*8)),seed=seed,reshuffle_each_iteration=True)
    def load_one(path,label):
        pixels=tf.py_function(_read_oriented_rgb,[path],Tout=tf.uint8); pixels.set_shape([None,None,3])
        return preprocess_rgb(pixels),label
    ds=ds.map(load_one,num_parallel_calls=1,deterministic=True).batch(batch_size,drop_remainder=False).prefetch(1)
    options=tf.data.Options(); options.deterministic=True; options.threading.private_threadpool_size=2; options.threading.max_intra_op_parallelism=1
    return ds.with_options(options)

def _read_oriented_rgb(path):
    from PIL import Image,ImageOps
    with Image.open(path.numpy().decode('utf-8')) as im: return np.asarray(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.uint8)

def make_model(keras,units,activation,name):
    inputs=keras.Input((224,224,3),name='rgb_image'); x=keras.layers.RandomFlip('horizontal')(inputs)
    x=keras.layers.RandomRotation(.08)(x); x=keras.layers.RandomZoom(.12)(x); x=keras.layers.RandomContrast(.12)(x)
    x=keras.applications.mobilenet_v2.preprocess_input(x*255.)
    backbone=keras.applications.MobileNetV2(include_top=False,weights='imagenet',input_tensor=x); backbone.trainable=False
    x=keras.layers.GlobalAveragePooling2D()(backbone.output); x=keras.layers.Dropout(.25)(x)
    return keras.Model(inputs,keras.layers.Dense(units,activation=activation,name=name)(x))

def fit_stage(tf,model,train_ds,val_ds,epochs,models,state,name):
    models.mkdir(exist_ok=True); state.mkdir(exist_ok=True)
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(models/f'{name}.best.keras'),monitor='val_accuracy',mode='max',save_best_only=True,verbose=1),tf.keras.callbacks.BackupAndRestore(backup_dir=str(state/name)),tf.keras.callbacks.CSVLogger(str(models/f'{name}.training.csv'),append=True)]
    return model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=callbacks,verbose=1)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data',type=Path,default=DEFAULT_ROOT); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--seed',type=int,default=SEED); args=ap.parse_args()
    if args.batch_size<1 or args.epochs<1: ap.error('--batch-size and --epochs must be positive')
    random.seed(args.seed); np.random.seed(args.seed)
    try: import tensorflow as tf
    except ImportError as exc: raise SystemExit('TensorFlow is required. Install with: python -m pip install -r requirements.txt') from exc
    samples=list(iter_samples(args.data)); leaves=[s for s in samples if s['leaf']]; negatives=sum(not s['leaf'] for s in samples)
    supported=[s for s in leaves if s['crop'] in CROPS]; missing=set(CROPS)-{s['crop'] for s in supported}
    print(f'Inventory: {len(leaves)} leaves, {negatives} non-leaf; {len(supported)} supported crop images; {len(leaves)-len(supported)} unsupported crop images excluded from crop/disease training')
    if missing: raise SystemExit('Cannot train all nine crops; missing images for: '+', '.join(sorted(missing)))
    if not negatives: raise SystemExit('No valid non-leaf images found.')
    root=Path(__file__).parent; models=root/'models'; state=root/'training_state'/'leaf-crop-disease-v2'; models.mkdir(exist_ok=True)
    gate_y=[int(s['leaf']) for s in samples]; ids=np.arange(len(samples)); gt,gv=train_test_split(ids,test_size=.2,random_state=args.seed,stratify=gate_y); paths=[s['path'] for s in samples]
    gtrain=make_dataset(tf,[paths[i] for i in gt],[gate_y[i] for i in gt],args.batch_size,True,args.seed); gval=make_dataset(tf,[paths[i] for i in gv],[gate_y[i] for i in gv],args.batch_size,False,args.seed)
    (models/'leaf_classes.json').write_text(json.dumps({'0':'Non-Leaf','1':'Leaf'},indent=2),encoding='utf-8')
    gate=make_model(tf.keras,1,'sigmoid','leaf_probability'); gate.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']); gh=fit_stage(tf,gate,gtrain,gval,args.epochs,models,state,'leaf_gate'); gate.save(models/'leaf_gate.keras')
    print(f"Stage 1 Leaf/Non-Leaf: train={len(gt)}, val={len(gv)}, accuracy={gh.history['accuracy'][-1]:.4f}, val_accuracy={gh.history['val_accuracy'][-1]:.4f}")

    crop_names=sorted(CROPS); crop_map={n:i for i,n in enumerate(crop_names)}; crop_y=[crop_map[s['crop']] for s in supported]; crop_paths=[s['path'] for s in supported]
    ci=np.arange(len(supported)); ct,cv=train_test_split(ci,test_size=.2,random_state=args.seed,stratify=crop_y)
    (models/'crop_classes.json').write_text(json.dumps(crop_names,indent=2),encoding='utf-8')
    crop_model=make_model(tf.keras,len(crop_names),'softmax','crop'); crop_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    ch=fit_stage(tf,crop_model,make_dataset(tf,[crop_paths[i] for i in ct],[crop_y[i] for i in ct],args.batch_size,True,args.seed),make_dataset(tf,[crop_paths[i] for i in cv],[crop_y[i] for i in cv],args.batch_size,False,args.seed),args.epochs,models,state,'crop_classifier'); crop_model.save(models/'crop_classifier.keras')
    print(f"Stage 2 crop: train={len(ct)}, val={len(cv)}, accuracy={ch.history['accuracy'][-1]:.4f}, val_accuracy={ch.history['val_accuracy'][-1]:.4f}")

    disease_names=sorted({s['crop']+'___'+s['disease'] for s in supported}); disease_map={n:i for i,n in enumerate(disease_names)}; disease_y=[disease_map[s['crop']+'___'+s['disease']] for s in supported]
    dt,dv=disease_split(disease_y,args.seed); (models/'disease_classes.json').write_text(json.dumps(disease_names,ensure_ascii=False,indent=2),encoding='utf-8')
    disease_model=make_model(tf.keras,len(disease_names),'softmax','crop_disease'); disease_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    dh=fit_stage(tf,disease_model,make_dataset(tf,[crop_paths[i] for i in dt],[disease_y[i] for i in dt],args.batch_size,True,args.seed),make_dataset(tf,[crop_paths[i] for i in dv],[disease_y[i] for i in dv],args.batch_size,False,args.seed),args.epochs,models,state,'disease_classifier'); disease_model.save(models/'disease_classifier.keras')
    print(f"Stage 3 disease: train={len(dt)}, val={len(dv)}, accuracy={dh.history['accuracy'][-1]:.4f}, val_accuracy={dh.history['val_accuracy'][-1]:.4f}, classes={len(disease_names)}")
    print(f'All three models and mappings saved in {models}')
if __name__=='__main__': main()

