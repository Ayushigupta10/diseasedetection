"""Use the saved leaf gate and disease classifier; crop is the disease-label prefix."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageOps,ImageStat
from image_preprocessing import preprocess_rgb
ROOT=Path(__file__).parent;MODELS=ROOT/'models'
NOT_LEAF='It is not a leaf. Please upload or capture a valid plant leaf image.'
UNCERTAIN='This may be a leaf, but it does not match a known crop and disease pattern confidently. No specific disease is reported. Try a clear photo of a supported crop leaf or consult an agricultural expert.'
def normalize_disease(name):return 'Healthy' if name.strip().lower() in {'healthy','healty'} else name.strip()
class Predictor:
 def __init__(self):
  self.gate=self.disease_model=None;self.gate_classes={};self.disease_classes=[];self.error=''
  try:
   import tensorflow as tf
   self.gate=tf.keras.models.load_model(self._model_path('leaf_gate'))
   self.disease_model=tf.keras.models.load_model(self._model_path('disease_classifier'))
   self.gate_classes=json.loads((MODELS/'leaf_classes.json').read_text(encoding='utf-8'))
   self.disease_classes=json.loads((MODELS/'disease_classes.json').read_text(encoding='utf-8'))
   if self.gate.output_shape[-1]!=1:raise ValueError('Expected leaf_gate.keras to have one sigmoid output.')
   if self.disease_model.output_shape[-1]!=len(self.disease_classes):raise ValueError('Disease model and disease_classes.json do not match.')
   if self.gate_classes.get('0')!='Non-Leaf' or self.gate_classes.get('1')!='Leaf':raise ValueError('leaf_classes.json must map 0=Non-Leaf and 1=Leaf.')
   self.crop_indexes={}
   for i,label in enumerate(self.disease_classes):
    crop,sep,disease=label.partition('___')
    if sep and crop and crop.lower()!='unknown':self.crop_indexes.setdefault(crop,[]).append(i)
   if not self.crop_indexes:raise ValueError('No crop prefixes found in disease_classes.json.')
  except Exception as exc:self.error=str(exc)
 @staticmethod
 def _model_path(name):
  final=MODELS/f'{name}.keras';best=MODELS/f'{name}.best.keras'
  if final.is_file():return final
  if best.is_file():return best
  raise FileNotFoundError(f'Missing trained model: {final.name}')
 @property
 def ready(self):return not self.error
 @staticmethod
 def _quality_uncertain(reason):
  return {'kind':'uncertain','leaf_validated':None,'crop':None,'disease':'Uncertain Disease','message':UNCERTAIN,'uncertainty_reason':reason}
 @staticmethod
 def _uncertain(crop='Unknown Crop',crop_confidence=None,reason=None,leaf_confidence=None):
  result={'kind':'uncertain','leaf_validated':True,'crop':None,'disease':'Uncertain Disease','message':UNCERTAIN}
  if crop_confidence is not None:result['crop_confidence']=float(crop_confidence)
  if leaf_confidence is not None:result['leaf_confidence']=float(leaf_confidence)
  if reason:result['uncertainty_reason']=reason
  return result
 def predict(self,image):
  if self.error:raise RuntimeError('The saved leaf and disease models are not ready. Check models/ and class JSON files.')
  image=ImageOps.exif_transpose(image).convert('RGB')
  if min(image.size)<80:return self._quality_uncertain('Image resolution is too small for a reliable check.')
  edge_std=ImageStat.Stat(image.convert('L').filter(ImageFilter.FIND_EDGES)).stddev[0]
  if edge_std<2.0:return self._quality_uncertain('The image appears too blurred or low contrast.')
  batch=preprocess_rgb(np.asarray(image,dtype=np.uint8))[None,...].numpy()
  # Stage 1: if not a leaf, stop before running disease classification.
  leaf_score=float(self.gate.predict(batch,verbose=0).reshape(-1)[0])
  # The leaf gate was trained with a very small non-leaf set. Require a strong
  # leaf score before exposing any disease result; borderline images are
  # rejected so arbitrary objects do not reach the disease classifier.
  leaf_threshold=float(os.getenv('LEAF_THRESHOLD','.995'))
  if leaf_score<leaf_threshold:
   # Low scores indicate a non-leaf. Borderline scores stay uncertain and
   # never proceed to crop or disease classification.
   uncertain_floor=float(os.getenv('UNCERTAIN_LEAF_FLOOR','.85'))
   if leaf_score<uncertain_floor:
    return {'kind':'not_leaf','leaf_validated':False,'leaf_confidence':leaf_score,'message':NOT_LEAF}
   result=self._quality_uncertain('The image does not match the trained leaf examples closely enough for a diagnosis.')
   result.update(leaf_validated=None,leaf_confidence=leaf_score,uncertainty_reason='Image is outside the confident leaf range.')
   return result  # The disease softmax contains crop___disease labels. Sum class probabilities
  # by crop prefix to obtain a visual crop score without a separate crop model.
  scores=self.disease_model.predict(batch,verbose=0)[0]
  crop_scores={crop:float(np.sum(scores[indexes])) for crop,indexes in self.crop_indexes.items()}
  ranked=sorted(crop_scores.items(),key=lambda pair:pair[1],reverse=True)
  crop,crop_score=ranked[0];crop_margin=crop_score-(ranked[1][1] if len(ranked)>1 else 0.0)
  if crop_score<float(os.getenv('CROP_THRESHOLD','.70')) or crop_margin<float(os.getenv('CROP_MARGIN','.20')):
   return self._uncertain(crop_confidence=crop_score,reason='Crop probability or margin is below threshold.',leaf_confidence=leaf_score)
  indexes=self.crop_indexes[crop];conditional=scores[indexes]/max(crop_score,1e-8)
  local=int(np.argmax(conditional));disease_score=float(conditional[local]);ordered=np.sort(conditional)
  disease_margin=float(ordered[-1]-(ordered[-2] if len(ordered)>1 else 0.0))
  if disease_score<float(os.getenv('DISEASE_THRESHOLD','.95')) or disease_margin<float(os.getenv('DISEASE_MARGIN','.20')):
   return self._uncertain(crop,crop_score,'Disease confidence or margin is below threshold.',leaf_score)
  disease=self.disease_classes[indexes[local]].partition('___')[2];disease=normalize_disease(disease)
  kind='healthy' if disease.lower()=='healthy' else 'disease'
  return {'kind':kind,'leaf_validated':True,'crop':crop,'disease':disease,'leaf_confidence':leaf_score,'crop_confidence':crop_score,'disease_confidence':disease_score,'confidence':disease_score}









