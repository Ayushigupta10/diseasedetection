"""Evaluate the saved leaf gate and disease classifier using streamed images."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from dataset import DEFAULT_ROOT,iter_samples
from train import make_dataset,disease_split,SEED

def canonical_label(label):
    crop,disease=label.split('___',1)
    if disease.lower()=='healty':disease='Healthy'
    return f'{crop}___{disease}'

def load_class_names(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if isinstance(value,dict):
        return [value[str(i)] for i in range(len(value))]
    if isinstance(value,list):return value
    raise ValueError(f'Unsupported class map format: {path}')

def report(name,actual,predicted,class_names,labels):
    print(f'\n{name} accuracy: {accuracy_score(actual,predicted):.4f}')
    print(classification_report(actual,predicted,labels=labels,target_names=class_names,zero_division=0,digits=4))
    print(f'{name} confusion matrix (true rows, predicted columns):')
    print(confusion_matrix(actual,predicted,labels=labels))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_ROOT,help='Folder containing the named crop datasets')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    try:import tensorflow as tf
    except ImportError as exc:raise SystemExit('TensorFlow is required. Activate the project venv and install requirements.txt.') from exc

    models=Path(__file__).parent/'models'
    gate_path=models/'leaf_gate.keras'; disease_path=models/'disease_classifier.keras'
    for path in (gate_path,disease_path):
        if not path.is_file():raise SystemExit(f'Required trained model is missing: {path}')
    gate_classes=load_class_names(models/'leaf_classes.json')
    disease_classes=load_class_names(models/'disease_classes.json')
    gate=tf.keras.models.load_model(gate_path); disease=tf.keras.models.load_model(disease_path)
    if gate.output_shape[-1]!=1:raise SystemExit('Expected leaf_gate.keras to have one sigmoid output.')
    if disease.output_shape[-1]!=len(disease_classes):raise SystemExit('Disease model output size does not match disease_classes.json.')
    gate_map={name.lower():i for i,name in enumerate(gate_classes)}
    if gate_map.get('leaf')!=1 or gate_map.get('non-leaf')!=0:raise SystemExit('leaf_classes.json must map 0=Non-Leaf and 1=Leaf for this sigmoid model.')

    # Dataset iteration verifies files, removes exact duplicates, and scans only named data roots.
    samples=list(iter_samples(args.data))
    gate_y=np.asarray([int(s['leaf']) for s in samples],dtype=np.int32)
    gate_ids=np.arange(len(samples))
    _,gate_val=train_test_split(gate_ids,test_size=.2,random_state=args.seed,stratify=gate_y)
    paths=[s['path'] for s in samples]
    gate_ds=make_dataset(tf,[paths[i] for i in gate_val],[gate_y[i] for i in gate_val],args.batch_size,False,args.seed)
    gate_scores=gate.predict(gate_ds,verbose=1).reshape(-1)
    gate_pred=(gate_scores>=.5).astype(np.int32)
    report('Leaf / non-leaf validation',gate_y[gate_val],gate_pred,gate_classes,[0,1])

    # The saved disease model has crop___disease class names. Sum output probabilities
    # by the crop prefix for crop identification; no crop_classifier.keras is needed.
    class_to_index={canonical_label(label):i for i,label in enumerate(disease_classes)}
    disease_samples=[]; disease_y=[]
    for sample in samples:
        if not sample['leaf']:continue
        key=canonical_label(sample['crop']+'___'+sample['disease'])
        if key in class_to_index:
            disease_samples.append(sample);disease_y.append(class_to_index[key])
    if not disease_samples:raise SystemExit('No dataset image classes match disease_classes.json.')
    disease_y=np.asarray(disease_y,dtype=np.int32)
    disease_train,disease_val=disease_split(disease_y.tolist(),args.seed)
    val_paths=[disease_samples[i]['path'] for i in disease_val]
    val_y=disease_y[disease_val]
    disease_ds=make_dataset(tf,val_paths,val_y,args.batch_size,False,args.seed)
    scores=disease.predict(disease_ds,verbose=1)

    crop_by_class=[canonical_label(label).split('___',1)[0] for label in disease_classes]
    crop_names=sorted({crop_by_class[i] for i in range(len(crop_by_class)) if crop_by_class[i]!='Unknown'})
    crop_indices={crop:[i for i,name in enumerate(crop_by_class) if name==crop] for crop in crop_names}
    crop_scores=np.column_stack([scores[:,crop_indices[crop]].sum(axis=1) for crop in crop_names])
    crop_pred=crop_scores.argmax(axis=1)
    true_crop_names=[crop_by_class[i] for i in val_y]
    crop_to_index={name:i for i,name in enumerate(crop_names)}
    true_crop=np.asarray([crop_to_index[name] for name in true_crop_names],dtype=np.int32)
    report('Crop identification (probability grouped by class prefix)',true_crop,crop_pred,crop_names,list(range(len(crop_names))))

    # Report disease accuracy when the crop is known, and end-to-end disease accuracy
    # after selecting the crop from the aggregated class probabilities.
    conditional_pred=[];end_to_end_pred=[]
    for row,scores_row in enumerate(scores):
        true_crop= true_crop_names[row]
        true_indices=crop_indices[true_crop]
        conditional_pred.append(max(true_indices,key=lambda i:scores_row[i]))
        predicted_crop=crop_names[int(crop_pred[row])]
        predicted_indices=crop_indices[predicted_crop]
        end_to_end_pred.append(max(predicted_indices,key=lambda i:scores_row[i]))
    class_labels=list(range(len(disease_classes)))
    present_classes=sorted(set(disease_y.tolist()))
    report('Disease (crop known)',val_y,np.asarray(conditional_pred),[disease_classes[i] for i in present_classes],present_classes)
    report('End-to-end crop + disease',val_y,np.asarray(end_to_end_pred),disease_classes,class_labels)
    absent=[name for name in disease_classes if canonical_label(name).split('___',1)[0] not in crop_names]
    if absent:print('\nModel outputs without matching crop folders (not evaluable from supplied crop data): '+', '.join(absent))
    print('\nNote: no training split manifest is stored with these models. This deterministic 80/20 split is a reconstructed evaluation split and cannot be guaranteed disjoint from the original training images.')

if __name__=='__main__':main()

